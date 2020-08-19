/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imagesegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import androidx.core.graphics.ColorUtils
import android.util.Log
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.collections.HashSet
import kotlin.random.Random
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
//import java.io.File
//import java.io.FileOutputStream

//import kotlin.math.abs

/**
 * Class responsible to run the Image Segmentation model.
 * more information about the DeepLab model being used can
 * be found here:
 * https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html
 * https://www.tensorflow.org/lite/models/segmentation/overview
 * https://github.com/tensorflow/models/tree/master/research/deeplab
 *
 * Label names: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 * 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 * 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
 */
class ImageSegmentationModelExecutor(
  context: Context,
  private var useGPU: Boolean = false
) {
  private var gpuDelegate: GpuDelegate? = null

  private val segmentationMasks: ByteBuffer
  private val interpreter: Interpreter

  private var fullTimeExecutionTime = 0L
  private var preprocessTime = 0L
  private var imageSegmentationTime = 0L
  private var maskFlatteningTime = 0L

  private var numberThreads = 4

  init {

    interpreter = getInterpreter(context, imageSegmentationModel, useGPU)
//    segmentationMasks = ByteBuffer.allocateDirect(1 * imageSize * imageSize *  * 4)
    segmentationMasks = ByteBuffer.allocateDirect(1 * imageSize * imageSize * NUM_CLASSES * 4)
    segmentationMasks.order(ByteOrder.nativeOrder())
  }

  fun execute(data: Bitmap): ModelExecutionResult {
    try {
      fullTimeExecutionTime = SystemClock.uptimeMillis()

      preprocessTime = SystemClock.uptimeMillis()
      val scaledBitmap =
        ImageUtils.scaleBitmapAndKeepRatio(
          data,
          imageSize, imageSize
        )

      val contentArray =
        ImageUtils.bitmapToByteBuffer(
          scaledBitmap,
          imageSize,
          imageSize,
          IMAGE_MEAN,
          IMAGE_STD
        )
      preprocessTime = SystemClock.uptimeMillis() - preprocessTime

      imageSegmentationTime = SystemClock.uptimeMillis()
      interpreter.run(contentArray, segmentationMasks)
      imageSegmentationTime = SystemClock.uptimeMillis() - imageSegmentationTime
      Log.d(TAG, "Time to run the model $imageSegmentationTime")

      maskFlatteningTime = SystemClock.uptimeMillis()
      val (maskImageApplied, maskOnly, itensFound) =
        convertBytebufferMaskToBitmap(
          segmentationMasks, imageSize, imageSize, scaledBitmap,
          segmentColors
        )
      maskFlatteningTime = SystemClock.uptimeMillis() - maskFlatteningTime
      Log.d(TAG, "Time to flatten the mask result $maskFlatteningTime")

      fullTimeExecutionTime = SystemClock.uptimeMillis() - fullTimeExecutionTime
      Log.d(TAG, "Total time execution $fullTimeExecutionTime")

      return ModelExecutionResult(
        maskImageApplied,
        scaledBitmap,
        maskOnly,
        formatExecutionLog(),
        itensFound
      )
    } catch (e: Exception) {
      val exceptionLog = "something went wrong: ${e.message}"
      Log.d(TAG, exceptionLog)

      val emptyBitmap =
        ImageUtils.createEmptyBitmap(
          imageSize,
          imageSize
        )
      return ModelExecutionResult(
        emptyBitmap,
        emptyBitmap,
        emptyBitmap,
        exceptionLog,
        HashSet(0)
      )
    }
  }

  // base: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java
  @Throws(IOException::class)
  private fun loadModelFile(context: Context, modelFile: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(modelFile)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    val retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    fileDescriptor.close()
    return retFile
  }

  @Throws(IOException::class)
  private fun getInterpreter(
    context: Context,
    modelName: String,
    useGpu: Boolean = false
  ): Interpreter {
    val tfliteOptions = Interpreter.Options()
    tfliteOptions.setNumThreads(numberThreads)

    gpuDelegate = null
    if (useGpu) {
      gpuDelegate = GpuDelegate()
      tfliteOptions.addDelegate(gpuDelegate)
    }

    return Interpreter(loadModelFile(context, modelName), tfliteOptions)
  }

  private fun formatExecutionLog(): String {
    val sb = StringBuilder()
    sb.append("Input Image Size: $imageSize x $imageSize\n")
    sb.append("GPU enabled: $useGPU\n")
    sb.append("Number of threads: $numberThreads\n")
    sb.append("Pre-process execution time: $preprocessTime ms\n")
    sb.append("Model execution time: $imageSegmentationTime ms\n")
    sb.append("Mask flatten time: $maskFlatteningTime ms\n")
    sb.append("Full execution time: $fullTimeExecutionTime ms\n")
    return sb.toString()
  }

  fun close() {
    interpreter.close()
    if (gpuDelegate != null) {
      gpuDelegate!!.close()
    }
  }

  private fun convertBytebufferMaskToBitmap(
    inputBuffer: ByteBuffer,
    imageWidth: Int,
    imageHeight: Int,
    backgroundImage: Bitmap,
    colors: IntArray
  ): Triple<Bitmap, Bitmap, Set<Int>> {
    val conf = Bitmap.Config.ARGB_8888
    val maskBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)
    val resultBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)
    val scaledBackgroundImage =
      ImageUtils.scaleBitmapAndKeepRatio(
        backgroundImage,
        imageWidth,
        imageHeight
      )
    val mSegmentBits = Array(imageWidth) { IntArray(imageHeight) }
    val itemsFound = HashSet<Int>()
    inputBuffer.rewind()

    for (y in 0 until imageHeight) {
      for (x in 0 until imageWidth) {
        mSegmentBits[x][y] = 0


//       minst get pixel.
//        val pixels = IntArray(width * height)
//        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
//
//        for (i in pixels.indices) {
//          // Set 0 for white and 255 for black pixels
//          val pixel = pixels[i]
//          // The color of the input is black so the blue channel will be 0xFF.
//          val channel = pixel and 0xff
//          inputBuffer.putFloat((0xff - channel).toFloat())

//          val value = (inputBuffer.getFloat((y * imageWidth  + x ) *4))*255
        for (c in 0 until  NUM_CLASSES) {
          val value = inputBuffer.getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c) * 4) * 255
//          System.out.printf("%f",value)
//          System.out.printf("%d,%d,%f",x,y, value)
          if (c==1 && value > 1.0) {
            mSegmentBits[x][y] = 1
          } else {
            mSegmentBits[x][y] = 0
          }
        }
//        System.out.printf("%d",mSegmentBits[x][y])
//          System.out.printf("%d,%d,%d", x, y, mSegmentBits[x][y])

          itemsFound.add(mSegmentBits[x][y])
          val newPixelColor = ColorUtils.compositeColors(
                  colors[mSegmentBits[x][y]],
                  scaledBackgroundImage.getPixel(x, y)
          )
          resultBitmap.setPixel(x, y, newPixelColor)
          maskBitmap.setPixel(x, y, colors[mSegmentBits[x][y]])

      }
    }
//    System.out.println(maskBitmap)
    return Triple(resultBitmap, maskBitmap, itemsFound)
  }

  companion object {

    private const val TAG = "ImageSegmentationMExec"
    private const val imageSegmentationModel = "myTestNAS.tflite"
    private const val imageSize = 300
    const val NUM_CLASSES = 2
    private const val IMAGE_MEAN = 0.0f
    private const val IMAGE_STD = 255.0f

    val segmentColors = IntArray(NUM_CLASSES)
    val labelsArrays = arrayOf(
            "background","foreground"
//            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
//            "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
//            "person", "potted plant", "sheep", "sofa", "train", "tv"
    )
//    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
//    "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
//    "person", "potted plant", "sheep", "sofa", "train", "tv"
    init {

      val random = Random(System.currentTimeMillis())
      segmentColors[0] = Color.TRANSPARENT
      for (i in 1 until NUM_CLASSES) {
        segmentColors[i] = Color.argb(
          (128),
          getRandomRGBInt(
            random
          ),
          getRandomRGBInt(
            random
          ),
          getRandomRGBInt(
            random
          )
        )
      }
    }

    private fun getRandomRGBInt(random: Random) = (255 * random.nextFloat()).toInt()
  }
}

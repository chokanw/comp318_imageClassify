package com.example.comp318imageclassify

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.support.image.TensorImage;

class MainActivity : AppCompatActivity() {

    private lateinit var selectBtn: Button
    private lateinit var predictBtn: Button
    private lateinit var captureBtn: Button
    private lateinit var result: TextView
    private lateinit var imageView: ImageView
    private var bitmap: Bitmap? = null

    private lateinit var tflite: Interpreter // TensorFlow Lite Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI components
        selectBtn = findViewById(R.id.selectBtn)
        predictBtn = findViewById(R.id.predictBtn)
        captureBtn = findViewById(R.id.captureBtn)
        result = findViewById(R.id.result)
        imageView = findViewById(R.id.imageView)

        // Load the TensorFlow Lite model
        tflite = Interpreter(loadModelFile())

        // Check permissions
        checkPermissions()

        // Set onClickListener for Select Image button
        selectBtn.setOnClickListener {
            openGallery()
        }

        // Set onClickListener for Predict button
        predictBtn.setOnClickListener {
            predictImage()
        }

        // Set onClickListener for Capture Image button
        captureBtn.setOnClickListener {
            captureImage()
        }
    }

    // Function to check runtime permissions
    private fun checkPermissions() {
        val permissions = arrayOf(
            Manifest.permission.CAMERA
        )

        permissions.forEach { permission ->
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, permissions, 0)
            }
        }
    }

    // Function to open the gallery and pick an image
    private fun openGallery() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            type = "image/*" // Restrict to image files
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        galleryLauncher.launch(intent)
    }

    // Function to capture an image using the camera
    private fun captureImage() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        cameraLauncher.launch(intent)
    }

    // Function to load the TensorFlow Lite model
    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = assets.openFd("mobilenet_v1_1.0_224_quant.tflite")
        val fileInputStream = assetFileDescriptor.createInputStream()
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Function to preprocess the image

    private fun preprocessImage(bitmap: Bitmap): TensorImage {
        val inputSize = 224 // Assuming 224x224 input size for MobileNet

        // Ensure the bitmap is in ARGB_8888 format
        val convertedBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888) {
            bitmap.copy(Bitmap.Config.ARGB_8888, true)
        } else {
            bitmap
        }

        // Create and load the bitmap into TensorImage
        val tensorImage = TensorImage()
        tensorImage.load(convertedBitmap)
        return tensorImage
    }


    // Function to perform image classification
    private fun predictImage() {
        if (bitmap != null) {
            try {
                // Preprocess the image
                val tensorImage = preprocessImage(bitmap!!)
                val inputBuffer = tensorImage.buffer // Get the ByteBuffer for the model input

                // Define the output buffer shape (adjust based on your model)
                val outputBuffer = Array(1) { FloatArray(1001) } // Example for MobileNet with 1001 classes

                // Run the TensorFlow Lite model
                tflite.run(inputBuffer, outputBuffer)

                // Get the index with the highest confidence
                val maxIndex = outputBuffer[0].indices.maxByOrNull { outputBuffer[0][it] } ?: -1
                val confidence = outputBuffer[0][maxIndex]

                // Display the result
                result.text = "Predicted Class: $maxIndex\nConfidence: $confidence"
            } catch (e: Exception) {
                result.text = "Error during inference: ${e.message}"
                e.printStackTrace()
            }
        } else {
            result.text = "Please select or capture an image first."
        }
    }


    // ActivityResultLauncher for handling image selection
    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { resultData ->
        if (resultData.resultCode == RESULT_OK && resultData.data != null) {
            val uri = resultData.data!!.data
            uri?.let {
                try {
                    bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                        val source = ImageDecoder.createSource(contentResolver, uri)
                        ImageDecoder.decodeBitmap(source)
                    } else {
                        @Suppress("DEPRECATION")
                        MediaStore.Images.Media.getBitmap(contentResolver, uri)
                    }
                    imageView.setImageBitmap(bitmap)
                    result.text = "Image Selected Successfully!"
                } catch (e: Exception) {
                    e.printStackTrace()
                    result.text = "Failed to load image."
                }
            }
        }
    }

    // ActivityResultLauncher for handling image capture
    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { resultData ->
        if (resultData.resultCode == RESULT_OK && resultData.data != null) {
            val extras = resultData.data!!.extras
            bitmap = extras?.get("data") as Bitmap?
            if (bitmap != null) {
                imageView.setImageBitmap(bitmap)
                result.text = "Image Captured Successfully!"
            } else {
                result.text = "Failed to capture image."
            }
        }
    }
}

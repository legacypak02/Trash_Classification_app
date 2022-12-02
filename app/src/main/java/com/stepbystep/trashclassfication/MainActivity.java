package com.stepbystep.trashclassfication;

import androidx.annotation.Nullable;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.stepbystep.trashclassfication.ml.Classifier;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView iv;
    TextView tv, conf;

    int imgSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ActionBar abar = getSupportActionBar();
        abar.hide();

        camera = findViewById(R.id.btnCamera);
        gallery = findViewById(R.id.btnChooseImg);
        iv = findViewById(R.id.imageView);
        tv = findViewById(R.id.tv_result);
        conf = findViewById(R.id.tv_conf);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                    Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(intent, 1);
                }else{
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, 5);
            }
        });
    }

    @SuppressLint("SetTextI18n")
    public void Classifier(Bitmap image) {
        try {
            Classifier model = Classifier.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imgSize * imgSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] values = new int[imgSize * imgSize];
            image.getPixels(values, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for (int i = 0; i < imgSize; i++) {
                for (int j = 0; j < imgSize; j++) {
                    int val = values[pixel++]; //RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f));
                }
                
            }
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Classifier.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidence = outputFeature0.getFloatArray();

            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidence.length; i++) {
                if (confidence[i] > maxConfidence) {
                    maxConfidence = confidence[i];
                    maxPos = i;
                }
            }
            String[] classes = {"박스", "유리", "캔류", "종이", "플라스틱", "일반쓰레기"};
            tv.setText(classes[maxPos]);
            conf.setText("예측이 맞을 확률:"+ String.valueOf(Math.floor(maxConfidence * 100))+"%");

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 1){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                iv.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imgSize, imgSize, false);
                Classifier(image);
            }else{
                Uri imgUri = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imgUri);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                iv.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imgSize, imgSize, false);
                Classifier(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
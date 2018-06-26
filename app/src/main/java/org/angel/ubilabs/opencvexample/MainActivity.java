package org.angel.ubilabs.opencvexample;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.speech.tts.TextToSpeech;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import org.angel.ubilabs.opencvexample.utils.PermissionUtils;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2, TextToSpeech.OnInitListener {

    /**
     * Initial & Debug
     */

    private static final String TAG = "MainActivity";
    private TextToSpeech tts;
    private String number_final;
    private ImageButton imageButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // 保持屏幕一直亮
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        // 语音播报
        tts = new TextToSpeech(this, this);
        imageButton = findViewById(R.id.imageButton);

        imageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                tts.speak(number_final, TextToSpeech.QUEUE_FLUSH, null);
            }
        });
    }

    // 语音播报
    public void onInit(int status) {
        // 判断是否转化成功
        if (status == TextToSpeech.SUCCESS) {
            //默认设定语言为英文，TTS 不支持中文。
            int result = tts.setLanguage(Locale.US);
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Toast.makeText(MainActivity.this, "TTS do not support this language! ", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (openCvCameraView != null) {
            openCvCameraView.disableView();
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
    }

    @Override
    protected void onResume() {
        super.onResume();
        doRequestPermission();
    }

    // 在 Android 6.0 中在manifest文件中定义权限之后还必须调用 checkSelfPermission 函数
    private void doRequestPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermission();
        } else {
            initCamera();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (openCvCameraView != null) {
            openCvCameraView.disableView();
        }
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
    }

    private void initCamera() {
        openCvCameraView = findViewById(R.id.HelloOpenCvView);
        openCvCameraView.setVisibility(SurfaceView.VISIBLE);
        openCvCameraView.setCvCameraViewListener(this);
        // 调整摄像头的大小（ 最宽640，最长480 ）
        openCvCameraView.setMaxFrameSize(640, 480);
        // 是否显示 FPS
        openCvCameraView.disableFpsMeter();
        openCvCameraView.enableView();
    }

    private void requestPermission() {
        PermissionUtils.requestMultiPermissions(this, mPermissionGrant);
    }

    private PermissionUtils.PermissionGrant mPermissionGrant = new PermissionUtils.PermissionGrant() {
        @Override
        public void onPermissionGranted(int requestCode) {
            switch (requestCode) {
                case PermissionUtils.CODE_CAMERA:
                    Toast.makeText(MainActivity.this, "Result Permission Grant CODE_CAMERA", Toast.LENGTH_SHORT).show();
                    break;
                case PermissionUtils.CODE_READ_EXTERNAL_STORAGE:
                    Toast.makeText(MainActivity.this, "Result Permission Grant CODE_READ_EXTERNAL_STORAGE", Toast.LENGTH_SHORT).show();
                    break;
                case PermissionUtils.CODE_WRITE_EXTERNAL_STORAGE:
                    Toast.makeText(MainActivity.this, "Result Permission Grant CODE_WRITE_EXTERNAL_STORAGE", Toast.LENGTH_SHORT).show();
                    break;
                default:
                    Toast.makeText(MainActivity.this, "Result Permission Grant CODE_MULTI_PERMISSION", Toast.LENGTH_SHORT).show();
                    break;
            }
        }
    };

    @Override
    public void onRequestPermissionsResult(final int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        PermissionUtils.requestPermissionsResult(this, requestCode, permissions, grantResults, mPermissionGrant);
        initCamera();
    }

    /**
     * OpenCV
     */
    private CameraBridgeViewBase openCvCameraView;
    private TextView displayResult;
    private TensorFlowInferenceInterface inferenceInterface;

    // 启动 openCV
    static {
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV not loaded");
        } else {
            Log.d(TAG, "OpenCV loaded");
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "file:///android_asset/mnist_op.pb");
        // textview
        displayResult = findViewById(R.id.displayResult);

    }

    @Override
    public void onCameraViewStopped() {
    }

    private Handler displayHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            String message = msg.getData().getString("result");
            String numbers[] = message.split("\n");
            String results[][] = new String[numbers.length][];
            double posibility[] = new double[numbers.length];
            int maxIndex = 0;
            for (int i = 0; i < numbers.length; i++) {
                results[i] = numbers[i].split(":");
                posibility[i] = Double.parseDouble(results[i][1]);
            }
            for (int j = 1; j < posibility.length; j++) {
                if (posibility[maxIndex] < posibility[j]) maxIndex = j;
            }

            String message_final;
            message_final = posibility[maxIndex] * 100 + " % of this is " + results[maxIndex][0];
            number_final = "This is " + results[maxIndex][0];
            displayResult.setText(message_final);
            super.handleMessage(msg);
        }
    };


    private void convertBitmap(Bitmap bitmap, int[] mImagePixels, float[] mImageData) {
        bitmap.getPixels(mImagePixels, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < bitmap.getWidth() * bitmap.getHeight(); i++) {
            mImageData[i] = convertToGreyScale(mImagePixels[i]);
        }
    }

    private float convertToGreyScale(int color) {
        return ((((color >> 16) & 0xFF) + ((color >> 8) & 0xFF) + (color & 0xFF)) / 3.0f -128)/ -128.0f;
    }

    private static int argmax(float[] probs) {
        int maxIdx = -1;
        float maxProb = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    int count=-1;
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgbaMat = inputFrame.rgba();

        count++;
        if(count%10!=0) return rgbaMat;

        int width = 640;
        int height = 480;
        int wCenter = width / 2;
        //Mat centerMat = new Mat();
        //   Mat centerMat = new Mat(rgbaMat, new Rect(new Point(wCenter - height / 2, 0), new Size(480, 480)));
        Mat centerMat = new Mat(rgbaMat, new Rect(new Point(wCenter - height / 2, 0), new Size(480, 480)));
//        List<Mat> mv = new ArrayList<>();
//        Core.split(mat, mv);
//        for (int i = 0; i < mat.channels(); i++) {
//            Imgproc.equalizeHist(mv.get(i), mv.get(i));
//        }
//        Core.merge(mv, centerMat);
//        double thresh = 127, maxval = 255;
//        int type = Imgproc.THRESH_BINARY;
//        Imgproc.threshold(mat, centerMat, thresh, maxval, type);
        Imgproc.resize(centerMat, centerMat, new Size(28, 28));
        Core.rectangle(rgbaMat, new Point(wCenter - height / 2, 0), new Point(wCenter + height / 2, height), new Scalar(255), 3);
        if (inferenceInterface != null) {
            Bitmap bmp = Bitmap.createBitmap(centerMat.width(), centerMat.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(centerMat, bmp);

            int[] intValues = new int[centerMat.width() * centerMat.height()];
            float[] floatValues = new float[centerMat.width() * centerMat.height()];
            bmp.getPixels(intValues, 0, bmp.getWidth(), 0, 0, bmp.getWidth(), bmp.getHeight());
            convertBitmap(bmp, intValues, floatValues);

            inferenceInterface.feed("x", floatValues, 1, 28, 28, 1);
            inferenceInterface.run(new String[]{"output"});
            float[] results = new float[10];
            inferenceInterface.fetch("output", results);

            String displayResult = "";
            int number = argmax(results);
            displayResult += number + ": " + results[number] + "\n";

            Message message = new Message();
            Bundle data = new Bundle();
            data.putString("result", displayResult);
            message.setData(data);
            displayHandler.sendMessage(message);

        }
        return rgbaMat;
    }


}
package facebook.f8demo;

import android.Manifest;
import android.app.ActionBar;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.util.Size;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import java.io.File;
import android.os.Environment;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;
import android.content.DialogInterface;
import android.app.AlertDialog;

import static android.view.View.SYSTEM_UI_FLAG_IMMERSIVE;

public final class  ClassifyCamera extends AppCompatActivity {
    private static final String TAG = "F8DEMO";
    private static final int REQUEST_CAMERA_PERMISSION = 200;

    private TextureView textureView;
    private String cameraId;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession cameraCaptureSessions;
    protected CaptureRequest.Builder captureRequestBuilder;
    private Size imageDimension;
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;
    private TextView tv;
    private String predictedClass = "none";
    private AssetManager mgr;
    private boolean processing = false;
    private Image image = null;
    private  int imgs_count = 0;
    private boolean run_HWC = false;


    static {
        System.loadLibrary("native-lib");
    }

    public native String classificationFromCaffe2(int h, int w, byte[] Y, byte[] U, byte[] V,
                                                  int rowStride, int pixelStride, boolean r_hwc);
    public native String classificationFromCaffe2t(int h, int w, byte[] Y, byte[] U, byte[] V,
                                                  int rowStride, int pixelStride, boolean r_hwc);
    public native void initCaffe2(AssetManager mgr);
    private class SetUpNeuralNetwork extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void[] v) {
            try {
                initCaffe2(mgr);
                predictedClass = "Neural net loaded! Inferring...";
            } catch (Exception e) {
                Log.d(TAG, "Couldn't load neural network.");
            }
            return null;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);

        mgr = getResources().getAssets();

        new SetUpNeuralNetwork().execute();

        View decorView = getWindow().getDecorView();
        int uiOptions = View.SYSTEM_UI_FLAG_FULLSCREEN;
        decorView.setSystemUiVisibility(uiOptions);

        setContentView(R.layout.activity_classify_camera);

        textureView = (TextureView) findViewById(R.id.textureView);
        textureView.setSystemUiVisibility(SYSTEM_UI_FLAG_IMMERSIVE);
        final GestureDetector gestureDetector = new GestureDetector(this.getApplicationContext(),
                new GestureDetector.SimpleOnGestureListener(){
            @Override
            public boolean onDoubleTap(MotionEvent e) {
                return true;
            }

            @Override
            public void onLongPress(MotionEvent e) {
                super.onLongPress(e);

            }

            @Override
            public boolean onDoubleTapEvent(MotionEvent e) {
                return true;
            }

            @Override
            public boolean onDown(MotionEvent e) {
                return true;
            }
        });

        textureView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                return gestureDetector.onTouchEvent(event);
            }
        });

        assert textureView != null;
        textureView.setSurfaceTextureListener(textureListener);
        tv = (TextView) findViewById(R.id.sample_text);

    }

    TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            //open your camera here
            openCamera();
        }
        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
            // Transform you image captured size according to the surface width and height
        }
        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }
        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        }
    };
    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            cameraDevice = camera;
            createCameraPreview();
        }
        @Override
        public void onDisconnected(CameraDevice camera) {
            cameraDevice.close();
        }
        @Override
        public void onError(CameraDevice camera, int error) {
            cameraDevice.close();
            cameraDevice = null;
        }
    };
    protected void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("Camera Background");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }
    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
    public Bitmap returnBitMap(String url){
        URL myFileUrl = null;
        Bitmap bitmap = null;
        try {
            myFileUrl = new URL(url);
        } catch (MalformedURLException e) {
            e.printStackTrace();
        }
        try {
            HttpURLConnection conn = (HttpURLConnection) myFileUrl
                    .openConnection();
            conn.setDoInput(true);
            conn.connect();
            InputStream is = conn.getInputStream();
            bitmap = BitmapFactory.decodeStream(is);
            is.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bitmap;
    }
    protected void createCameraPreview() {
        try {
            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
            Surface surface = new Surface(texture);
//            int width = 227;
//            int height = 227;
            int width = 1024;
            int height = 1024;

            //ImageReader reader = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 4);
            ImageReader reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 5);

            ImageReader.OnImageAvailableListener readerListener = new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {

                    //
                    FileOutputStream fos = null;
                    Bitmap bitmap = null;
                    try {
                       // Bitmap.CompressFormat.PNG

                        //image = reader.acquireNextImage();
                        image = reader.acquireLatestImage();
                        if (processing) {
                            image.close();
                            return;
                        }
                        processing = true;
                        if (true) {
                            Image.Plane[] planes = image.getPlanes();
                            if (planes[0].getBuffer() == null) {
                                processing = false;
                                return;
                            }
                            int width = image.getWidth();
                            int height = image.getHeight();
                            int pixelStride = planes[0].getPixelStride();
                            int rowStride = planes[0].getRowStride();
                            int rowPadding = rowStride - pixelStride * width;
                            byte[] newData = new byte[width * height * 4];

                            int offset = 0;
                            //Bitmap createBitmap(DisplayMetrics display, int width, int height, Bitmap.Config config)
                            //bitmap = Bitmap.createBitmap(metrics,width, height, Bitmap.Config.ARGB_8888);
                            bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                            ByteBuffer buffer = planes[0].getBuffer();
                            for (int i = 0; i < height-1; ++i) {
                                for (int j = 0; j < width; ++j) {
                                    int pixel = 0;
                                    pixel |= (buffer.get(offset) & 0xff) << 16;     // R
                                    pixel |= (buffer.get(offset + 1) & 0xff) << 8;  // G
                                    pixel |= (buffer.get(offset + 2) & 0xff);       // B
                                    pixel |= (buffer.get(offset + 3) & 0xff) << 24; // A
                                    bitmap.setPixel(j, i, pixel);
                                    offset += pixelStride;
                                }
                                offset += rowPadding;
                            }
                            String name = "/111111/test/" + imgs_count + ".png";
                            imgs_count++;
                            File file = new File(Environment.getExternalStorageDirectory(), name);
                            fos = new FileOutputStream(file);
                            //bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
                            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
                            //Log.i(TAG, "image saved in" + Environment.getExternalStorageDirectory() + name);
//                            predictedClass = classificationFromCaffe2t(h, w, Y, U, V,
//                                    rowStride, pixelStride, run_HWC);
                        }
                        if(!true){
                            int w = image.getWidth();
                            int h = image.getHeight();
                            ByteBuffer Ybuffer = image.getPlanes()[0].getBuffer();
                            ByteBuffer Ubuffer = image.getPlanes()[1].getBuffer();
                            ByteBuffer Vbuffer = image.getPlanes()[2].getBuffer();
                            // TODO: use these for proper image processing on different formats.
                            int rowStride = image.getPlanes()[1].getRowStride();
                            int pixelStride = image.getPlanes()[1].getPixelStride();
                            byte[] Y = new byte[Ybuffer.capacity()];
                            byte[] U = new byte[Ubuffer.capacity()];
                            byte[] V = new byte[Vbuffer.capacity()];
                            Ybuffer.get(Y);
                            Ubuffer.get(U);
                            Vbuffer.get(V);

                            predictedClass = classificationFromCaffe2(h, w, Y, U, V, rowStride, pixelStride, run_HWC);
                            //predictedClass = classificationFromCaffe2t(h, w, Y, U, V, rowStride, pixelStride, run_HWC);
                        }
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                tv.setText(predictedClass);
                            }
                        });
/*
                        String[] list=predictedClass.split("\n");
                        list[0]="以下都不是";
                        new AlertDialog.Builder(ClassifyCamera.this)
                            .setTitle("是否以下物品？")
                            .setIcon(android.R.drawable.ic_dialog_info)
                            .setSingleChoiceItems(list,//new String[] {"选项1","选项2","选项3","选项4"}
                                0,
                                new DialogInterface.OnClickListener() {
                                    public void onClick(DialogInterface dialog, int which) {
                                        dialog.dismiss();
                                            processing = false;
                                        }
                                    }
                                )
                            .setNegativeButton("确定",
                                new DialogInterface.OnClickListener() {
                                    public void onClick(DialogInterface dialog, int which) {
                                        dialog.dismiss();
                                        processing = false;
                                    }
                                }
                            )
                            .show();
                        //*/
//                        Handler h1= new Handler();
//                        h1.postDelayed(new Runnable(){
//                            public void run() {
//                                processing = false;
//                            }
//                        }, 10000);
                    } catch (Exception e) {
                        e.printStackTrace();
                    } finally {
                        if (image != null) {
                            image.close();
                        }
                        if (null != fos) {
                            try {
                                fos.close();
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                        }
                        if (null != bitmap) {
                            bitmap.recycle();
                        }
                        processing = false;
                    }
                }
            };
            reader.setOnImageAvailableListener(readerListener, mBackgroundHandler);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);
            captureRequestBuilder.addTarget(reader.getSurface());

            cameraDevice.createCaptureSession(Arrays.asList(surface, reader.getSurface()), new CameraCaptureSession.StateCallback(){
                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    if (null == cameraDevice) {
                        return;
                    }
                    cameraCaptureSessions = cameraCaptureSession;
                    updatePreview();
                }
                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Toast.makeText(ClassifyCamera.this, "Configuration change", Toast.LENGTH_SHORT).show();
                }
            }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }
    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[0];
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(ClassifyCamera.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                return;
            }
            manager.openCamera(cameraId, stateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    protected void updatePreview() {
        if(null == cameraDevice) {
            return;
        }
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        try {
            cameraCaptureSessions.setRepeatingRequest(captureRequestBuilder.build(), null, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void closeCamera() {
        if (null != cameraDevice) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(ClassifyCamera.this, "You can't use this app without granting permission", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        if (textureView.isAvailable()) {
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(textureListener);
        }
    }

    @Override
    protected void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }
}

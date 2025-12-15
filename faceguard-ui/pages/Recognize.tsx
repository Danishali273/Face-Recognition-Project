import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Camera, Scan, AlertOctagon } from 'lucide-react';
import { getTrainingStatus, VideoRecognitionSocket, canvasToBase64 } from '../services/faceService';
import { Link } from 'react-router-dom';

interface DetectedFace {
  name: string;
  confidence: number;
  bbox: number[]; // [top, right, bottom, left]
}

const Recognize: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<VideoRecognitionSocket | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelReady, setModelReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([]);
  const [fps, setFps] = useState(0);

  // Check model status on mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const status = await getTrainingStatus();
        setModelReady(status.isTrained);
      } catch (err) {
        console.error('Error checking model status:', err);
      } finally {
        setIsLoading(false);
      }
    };
    checkStatus();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (err) {
      setError("Camera access denied.");
    }
  };

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (modelReady && !isLoading) {
      startCamera();
    }
    return () => stopCamera();
  }, [modelReady, isLoading, stopCamera]);

  // WebSocket for real-time recognition
  useEffect(() => {
    if (!isStreaming || !modelReady) return;

    // Connect to WebSocket
    socketRef.current = new VideoRecognitionSocket();
    socketRef.current.connect((faces) => {
      setDetectedFaces(faces);
    });

    let lastTime = performance.now();
    let frameCount = 0;

    // Send frames to backend
    const sendFrame = () => {
      if (!videoRef.current || !canvasRef.current || !socketRef.current) return;
      
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      ctx.drawImage(video, 0, 0);
      const imageData = canvasToBase64(canvas);
      socketRef.current.sendFrame(imageData);
      
      // Calculate FPS
      frameCount++;
      const now = performance.now();
      if (now - lastTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastTime = now;
      }
    };

    const interval = setInterval(sendFrame, 200); // 5 FPS to backend

    return () => {
      clearInterval(interval);
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [isStreaming, modelReady]);

  if (isLoading) {
    return (
      <div className="h-full flex flex-col items-center justify-center space-y-4">
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-gray-400">Checking model status...</p>
      </div>
    );
  }

  if (!modelReady) {
    return (
      <div className="h-full flex flex-col items-center justify-center space-y-4">
        <div className="p-4 bg-amber-500/10 rounded-full">
          <AlertOctagon className="w-12 h-12 text-amber-500" />
        </div>
        <h2 className="text-2xl font-bold text-white">Model Not Ready</h2>
        <p className="text-gray-400 max-w-md text-center">
          You need to train the model before using live recognition. 
          Please ensure you have captured face data and run the training process.
        </p>
        <Link 
          to="/train" 
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
        >
          Go to Training
        </Link>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Hidden canvas for capturing frames */}
      <canvas ref={canvasRef} className="hidden" />
      
      <div className="flex justify-between items-center mb-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Live Recognition</h1>
          <p className="text-gray-400">Real-time inference using the trained KNN model.</p>
        </div>
        {isStreaming && (
          <div className="flex items-center space-x-2 px-3 py-1 bg-red-500/20 border border-red-500/40 rounded-full">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-xs text-red-400 font-bold tracking-wider">LIVE</span>
          </div>
        )}
      </div>

      <div className="flex-1 bg-black rounded-xl overflow-hidden border border-gray-800 relative shadow-2xl">
        {error ? (
          <div className="absolute inset-0 flex items-center justify-center text-red-400">
            {error}
          </div>
        ) : (
          <>
            <video 
              ref={videoRef}
              autoPlay 
              playsInline 
              muted
              className="w-full h-full object-cover transform scale-x-[-1]"
            />
            
            {/* HUD Overlay */}
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-transparent via-green-500/50 to-transparent opacity-50 scan-line"></div>
              
              {/* Detected Face Bounding Boxes */}
              {detectedFaces.map((face, index) => {
                // bbox is [top, right, bottom, left] - scale to video size
                const videoWidth = videoRef.current?.videoWidth || 640;
                const videoHeight = videoRef.current?.videoHeight || 480;
                const displayWidth = videoRef.current?.clientWidth || 640;
                const displayHeight = videoRef.current?.clientHeight || 480;
                
                const scaleX = displayWidth / videoWidth;
                const scaleY = displayHeight / videoHeight;
                
                const top = face.bbox[0] * scaleY;
                const right = (videoWidth - face.bbox[1]) * scaleX; // Flip for mirrored video
                const bottom = face.bbox[2] * scaleY;
                const left = (videoWidth - face.bbox[3]) * scaleX; // Flip for mirrored video
                
                const width = Math.abs(right - left);
                const height = bottom - top;
                
                return (
                  <div 
                    key={index}
                    className="absolute border-2 border-green-500 shadow-[0_0_20px_rgba(34,197,94,0.4)] transition-all duration-150"
                    style={{
                      top: `${top}px`,
                      left: `${Math.min(left, right)}px`,
                      width: `${width}px`,
                      height: `${height}px`,
                    }}
                  >
                    {/* Corners */}
                    <div className="absolute -top-1 -left-1 w-4 h-4 border-t-4 border-l-4 border-green-500"></div>
                    <div className="absolute -top-1 -right-1 w-4 h-4 border-t-4 border-r-4 border-green-500"></div>
                    <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-4 border-l-4 border-green-500"></div>
                    <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-4 border-r-4 border-green-500"></div>

                    {/* Label */}
                    <div className="absolute -bottom-10 left-1/2 -translate-x-1/2 bg-green-600/90 text-white px-3 py-1.5 rounded text-sm font-mono whitespace-nowrap backdrop-blur-sm">
                      <span className="font-bold">{face.name}</span>
                      <span className="ml-2 opacity-75">{(face.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                );
              })}
              
              {/* HUD Elements */}
              <div className="absolute bottom-6 left-6 text-green-500/60 font-mono text-xs space-y-1">
                <p>FPS: {fps}</p>
                <p>FACES: {detectedFaces.length}</p>
                <p>MODEL: KNN (k=5)</p>
              </div>
              
              <div className="absolute top-6 left-6 text-green-500/60">
                 <Scan className="w-8 h-8 opacity-50" />
              </div>
            </div>
          </>
        )}
        
        {!isStreaming && !error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500">
            <Camera className="w-16 h-16 mb-4 opacity-20" />
            <p>Initializing Camera Feed...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Recognize;
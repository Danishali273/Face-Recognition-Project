import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Camera, StopCircle, PlayCircle, RefreshCw, CheckCircle2 } from 'lucide-react';
import { saveFace, resetModelStatus, canvasToBase64 } from '../services/faceService';

const Capture: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [name, setName] = useState('');
  const [maxSamples, setMaxSamples] = useState(15);
  const [currentSamples, setCurrentSamples] = useState(0);
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [completed, setCompleted] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string>('');

  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (err) {
      console.error(err);
      setError("Unable to access camera. Please ensure you have granted permission.");
    }
  };

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  }, []);

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  const captureFrame = (): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    
    ctx.drawImage(video, 0, 0);
    return canvasToBase64(canvas);
  };

  const handleCaptureSession = async () => {
    if (!name.trim()) return;
    setIsCapturing(true);
    setCurrentSamples(0);
    setCompleted(false);
    setStatusMessage('Starting capture...');

    let successCount = 0;
    let failCount = 0;
    const maxAttempts = maxSamples * 3; // Allow extra attempts for failed captures
    
    for (let attempt = 0; attempt < maxAttempts && successCount < maxSamples; attempt++) {
      const imageData = captureFrame();
      if (!imageData) {
        await new Promise(r => setTimeout(r, 200));
        continue;
      }

      try {
        const result = await saveFace(name, imageData);
        
        if (result.success) {
          successCount++;
          setCurrentSamples(successCount);
          setStatusMessage(`Captured ${successCount}/${maxSamples}`);
        } else {
          failCount++;
          setStatusMessage(result.message || 'Capture failed, retrying...');
        }
      } catch (err) {
        console.error('Capture error:', err);
        setStatusMessage('API error, retrying...');
      }

      // Wait between captures
      await new Promise(r => setTimeout(r, 300));
    }

    setIsCapturing(false);
    
    if (successCount >= maxSamples) {
      setCompleted(true);
      resetModelStatus();
      setStatusMessage('Capture complete!');
      stopCamera();
    } else {
      setError(`Only captured ${successCount}/${maxSamples} samples. Please try again with better lighting.`);
    }
  };

  const resetForm = () => {
    setCompleted(false);
    setCurrentSamples(0);
    setName('');
    setError(null);
    setStatusMessage('');
    startCamera();
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Hidden canvas for capturing frames */}
      <canvas ref={canvasRef} className="hidden" />
      
      <div>
        <h1 className="text-2xl font-bold text-white mb-2">Capture New Face</h1>
        <p className="text-gray-400">Add a new identity to the dataset. Position the subject in the frame.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Controls Column */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">Person Name</label>
              <input 
                type="text" 
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="John Doe"
                className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-green-500 focus:outline-none"
                disabled={isCapturing || completed}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">Samples Required</label>
              <select 
                value={maxSamples}
                onChange={(e) => setMaxSamples(Number(e.target.value))}
                className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-green-500 focus:outline-none"
                disabled={isCapturing || completed}
              >
                <option value={15}>15 (Fast)</option>
                <option value={30}>30 (Balanced)</option>
                <option value={50}>50 (High Accuracy)</option>
              </select>
            </div>

            <div className="pt-4">
               {!isStreaming && !completed ? (
                  <button 
                    onClick={startCamera}
                    className="w-full flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition-colors"
                  >
                    <Camera className="w-5 h-5" />
                    <span>Start Camera</span>
                  </button>
               ) : !completed ? (
                  isCapturing ? (
                    <button 
                      disabled
                      className="w-full flex items-center justify-center space-x-2 bg-gray-700 text-gray-400 font-medium py-3 rounded-lg cursor-not-allowed"
                    >
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      <span>Capturing... {Math.round((currentSamples / maxSamples) * 100)}%</span>
                    </button>
                  ) : (
                    <button 
                      onClick={handleCaptureSession}
                      disabled={!name.trim()}
                      className={`w-full flex items-center justify-center space-x-2 font-medium py-3 rounded-lg transition-colors ${
                        name.trim() 
                          ? 'bg-green-600 hover:bg-green-700 text-white' 
                          : 'bg-gray-800 text-gray-500 cursor-not-allowed'
                      }`}
                    >
                      <PlayCircle className="w-5 h-5" />
                      <span>Start Capture</span>
                    </button>
                  )
               ) : (
                 <button 
                   onClick={resetForm}
                   className="w-full flex items-center justify-center space-x-2 bg-gray-800 hover:bg-gray-700 text-white font-medium py-3 rounded-lg transition-colors"
                 >
                   <RefreshCw className="w-5 h-5" />
                   <span>Add Another</span>
                 </button>
               )}
            </div>
          </div>

          {/* Guidelines */}
          <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-800/50">
            <h3 className="font-semibold text-gray-300 mb-3">Guidelines</h3>
            <ul className="text-sm text-gray-400 space-y-2 list-disc list-inside">
              <li>Ensure good lighting on the face.</li>
              <li>Remove accessories like sunglasses.</li>
              <li>Rotate head slightly during capture.</li>
              <li>Keep neutral expression initially.</li>
            </ul>
          </div>
        </div>

        {/* Video Feed Column */}
        <div className="lg:col-span-2">
          <div className="relative aspect-video bg-black rounded-xl overflow-hidden border-2 border-gray-800 shadow-2xl">
            {error ? (
              <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-red-400">
                {error}
              </div>
            ) : completed ? (
               <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4 bg-gray-900">
                 <div className="w-20 h-20 rounded-full bg-green-500/20 flex items-center justify-center">
                    <CheckCircle2 className="w-10 h-10 text-green-500" />
                 </div>
                 <h3 className="text-xl font-bold text-white">Capture Complete</h3>
                 <p className="text-gray-400">Data for <span className="text-white font-semibold">{name}</span> has been saved.</p>
               </div>
            ) : (
              <>
                <video 
                  ref={videoRef}
                  autoPlay 
                  playsInline 
                  muted
                  className={`w-full h-full object-cover transform scale-x-[-1] ${!isStreaming ? 'hidden' : ''}`}
                />
                
                {/* Overlay UI */}
                {isStreaming && (
                  <div className="absolute inset-0 pointer-events-none">
                    {/* Face Guide Box */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 border-2 border-white/30 rounded-full"></div>
                    
                    {/* Scanning Line if Capturing */}
                    {isCapturing && (
                      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 border-2 border-green-500 rounded-lg">
                        <div className="absolute inset-x-0 h-1 bg-green-500/50 animate-pulse top-0 shadow-[0_0_15px_rgba(34,197,94,0.5)] transition-all duration-1000" 
                             style={{ top: `${(currentSamples / maxSamples) * 100}%` }}
                        />
                        <div className="absolute -bottom-8 left-0 right-0 text-center">
                          <span className="bg-green-500 text-black text-xs font-bold px-2 py-1 rounded">
                            CAPTURING
                          </span>
                        </div>
                      </div>
                    )}

                    <div className="absolute top-4 right-4 bg-black/60 px-3 py-1 rounded-full text-xs font-mono text-green-400 flex items-center space-x-2">
                      <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                      <span>LIVE FEED {isCapturing ? `• ${currentSamples}/${maxSamples}` : ''}</span>
                    </div>
                  </div>
                )}

                {!isStreaming && !completed && (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                    <div className="text-center">
                      <Camera className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>Camera is off</p>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Capture;
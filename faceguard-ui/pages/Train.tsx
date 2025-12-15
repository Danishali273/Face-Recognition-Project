import React, { useState, useEffect } from 'react';
import { BrainCircuit, CheckCircle2, AlertTriangle, Play } from 'lucide-react';
import { getStoredFaces, trainModel, getTrainingStatus } from '../services/faceService';
import { TrainingStatus } from '../types';

const Train: React.FC = () => {
  const [status, setStatus] = useState<TrainingStatus>({ isTrained: false, lastTrainedAt: null, accuracy: 0 });
  const [isTraining, setIsTraining] = useState(false);
  const [hasData, setHasData] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    refreshStatus();
  }, []);

  const refreshStatus = async () => {
    try {
      const s = await getTrainingStatus();
      setStatus(s);
      const faces = await getStoredFaces();
      setHasData(faces.length > 0);
    } catch (err) {
      console.error('Error refreshing status:', err);
    }
  };

  const addLog = (msg: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  const handleTrain = async () => {
    setIsTraining(true);
    setLogs([]);
    setError(null);
    
    addLog("Initializing training sequence...");
    await new Promise(r => setTimeout(r, 500));
    
    addLog("Loading face embeddings from database...");
    await new Promise(r => setTimeout(r, 500));
    
    addLog("Configuring KNN Classifier (k=5, metric='euclidean')...");
    
    try {
      addLog("Sending training request to API server...");
      const newStatus = await trainModel();
      setStatus(newStatus);
      addLog("Training complete.");
      addLog(`Model saved successfully!`);
      if ((newStatus as any).classes) {
        addLog(`Classes learned: ${(newStatus as any).classes.join(', ')}`);
      }
      addLog(`Accuracy: ${(newStatus.accuracy * 100).toFixed(1)}%`);
    } catch (e: any) {
      addLog(`Error: ${e.message || 'Training failed'}`);
      setError(e.message || 'Training failed');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div className="text-center space-y-4">
        <div className="w-16 h-16 bg-blue-500/10 rounded-full flex items-center justify-center mx-auto">
          <BrainCircuit className="w-8 h-8 text-blue-500" />
        </div>
        <h1 className="text-3xl font-bold text-white">Model Training</h1>
        <p className="text-gray-400 max-w-lg mx-auto">
          Train the K-Nearest Neighbors (KNN) classifier on the collected face embeddings to enable recognition.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Status Card */}
        <div className="bg-gray-900 border border-gray-800 p-6 rounded-xl relative overflow-hidden">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Current Status</h3>
          
          <div className="flex items-center space-x-4 mb-6">
            {status.isTrained ? (
              <div className="bg-green-500/20 p-3 rounded-full">
                <CheckCircle2 className="w-8 h-8 text-green-500" />
              </div>
            ) : (
              <div className="bg-amber-500/20 p-3 rounded-full">
                <AlertTriangle className="w-8 h-8 text-amber-500" />
              </div>
            )}
            <div>
              <p className={`text-xl font-bold ${status.isTrained ? 'text-green-400' : 'text-amber-400'}`}>
                {status.isTrained ? 'Model Ready' : 'Training Required'}
              </p>
              <p className="text-sm text-gray-500">
                {status.lastTrainedAt ? `Last trained: ${new Date(status.lastTrainedAt).toLocaleString()}` : 'Never trained'}
              </p>
            </div>
          </div>

          <button
            onClick={handleTrain}
            disabled={isTraining || !hasData}
            className={`
              w-full py-4 rounded-lg font-bold flex items-center justify-center space-x-2 transition-all
              ${!hasData 
                ? 'bg-gray-800 text-gray-500 cursor-not-allowed' 
                : isTraining 
                  ? 'bg-blue-600/50 text-blue-200 cursor-wait' 
                  : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20'}
            `}
          >
            {isTraining ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Training Model...</span>
              </>
            ) : (
              <>
                <Play className="w-5 h-5 fill-current" />
                <span>Start Training</span>
              </>
            )}
          </button>
          
          {!hasData && (
             <p className="text-xs text-red-400 mt-3 text-center">No face data found. Please capture faces first.</p>
          )}
        </div>

        {/* Console Log Simulation */}
        <div className="bg-black border border-gray-800 p-4 rounded-xl font-mono text-xs h-64 overflow-y-auto flex flex-col">
          <div className="sticky top-0 bg-black pb-2 border-b border-gray-900 mb-2 text-gray-500 font-bold flex justify-between">
            <span>TERMINAL OUTPUT</span>
            <div className="flex space-x-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-red-500/50"></div>
              <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50"></div>
              <div className="w-2.5 h-2.5 rounded-full bg-green-500/50"></div>
            </div>
          </div>
          <div className="flex-1 space-y-1">
            {logs.length === 0 && (
              <span className="text-gray-700 italic">Waiting for process start...</span>
            )}
            {logs.map((log, i) => (
              <div key={i} className="text-green-400 animate-fade-in-up">
                <span className="text-gray-600 mr-2">$</span>
                {log}
              </div>
            ))}
            {isTraining && (
               <div className="text-blue-400 animate-pulse">
                 <span className="text-gray-600 mr-2">$</span>
                 _
               </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Train;
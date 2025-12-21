import { FaceData, TrainingStatus } from '../types';

// API Base URL - Change this to your Python backend URL
const API_BASE_URL = 'http://localhost:8000';

// Helper to handle API errors
const handleResponse = async (response: Response) => {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || 'API request failed');
  }
  return response.json();
};

// ============== FACE DATA OPERATIONS ==============

export const getStoredFaces = async (): Promise<FaceData[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/faces`);
    return handleResponse(response);
  } catch (error) {
    console.error('Error fetching faces:', error);
    // Fallback to localStorage if API is down
    const data = localStorage.getItem('face_app_data');
    return data ? JSON.parse(data) : [];
  }
};

export const saveFace = async (name: string, imageData: string): Promise<{ success: boolean; message: string; sampleCount: number }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/capture`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, image: imageData }),
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Error saving face:', error);
    throw error;
  }
};

export const deleteFace = async (name: string): Promise<void> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/faces/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
    await handleResponse(response);
  } catch (error) {
    console.error('Error deleting face:', error);
    throw error;
  }
};

// ============== MODEL TRAINING ==============

export const getTrainingStatus = async (): Promise<TrainingStatus> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/training/status`);
    return handleResponse(response);
  } catch (error) {
    console.error('Error fetching training status:', error);
    return { isTrained: false, lastTrainedAt: null, accuracy: 0 };
  }
};

export const trainModel = async (): Promise<TrainingStatus> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/training/train`, {
      method: 'POST',
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Error training model:', error);
    throw error;
  }
};

// ============== FACE RECOGNITION ==============

export const recognizeFace = async (imageData: string): Promise<{ faces: Array<{ name: string; confidence: number; bbox: number[] }> }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/recognize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: '', image: imageData }),
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Error recognizing face:', error);
    throw error;
  }
};

// ============== WEBSOCKET FOR REAL-TIME VIDEO ==============

export class VideoRecognitionSocket {
  private ws: WebSocket | null = null;
  private onResult: ((faces: Array<{ name: string; confidence: number; bbox: number[] }>) => void) | null = null;

  connect(onResult: (faces: Array<{ name: string; confidence: number; bbox: number[] }>) => void) {
    this.onResult = onResult;
    this.ws = new WebSocket(`ws://localhost:8000/ws/video`);
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (this.onResult) {
        this.onResult(data.faces || []);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket closed');
    };
  }

  sendFrame(imageData: string) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'frame', image: imageData }));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// Legacy function for backwards compatibility
export const resetModelStatus = () => {
  // No-op - model status is now managed by backend
};

// Helper to convert canvas to base64
export const canvasToBase64 = (canvas: HTMLCanvasElement): string => {
  return canvas.toDataURL('image/jpeg', 0.8);
};
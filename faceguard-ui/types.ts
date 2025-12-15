export interface FaceData {
  id: string;
  name: string;
  sampleCount: number;
  dateAdded: string;
  lastUpdated: string;
}

export interface TrainingStatus {
  isTrained: boolean;
  lastTrainedAt: string | null;
  accuracy: number;
}

export enum AppRoute {
  DASHBOARD = '/',
  CAPTURE = '/capture',
  MANAGE = '/manage',
  TRAIN = '/train',
  RECOGNIZE = '/recognize',
}
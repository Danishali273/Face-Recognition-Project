import React, { useEffect, useState } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';
import { Users, Database, Activity, AlertCircle } from 'lucide-react';
import { getStoredFaces, getTrainingStatus } from '../services/faceService';
import { FaceData, TrainingStatus } from '../types';

const Dashboard: React.FC = () => {
  const [faces, setFaces] = useState<FaceData[]>([]);
  const [modelStatus, setModelStatus] = useState<TrainingStatus>({ isTrained: false, lastTrainedAt: null, accuracy: 0 });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [facesData, statusData] = await Promise.all([
          getStoredFaces(),
          getTrainingStatus()
        ]);
        setFaces(facesData);
        setModelStatus(statusData);
      } catch (error) {
        console.error('Error loading dashboard data:', error);
      } finally {
        setIsLoading(false);
      }
    };
    loadData();
  }, []);

  const totalSamples = faces.reduce((acc, curr) => acc + curr.sampleCount, 0);

  const stats = [
    { 
      label: 'Total Identities', 
      value: faces.length, 
      icon: Users, 
      color: 'text-blue-400', 
      bg: 'bg-blue-400/10' 
    },
    { 
      label: 'Total Samples', 
      value: totalSamples, 
      icon: Database, 
      color: 'text-purple-400', 
      bg: 'bg-purple-400/10' 
    },
    { 
      label: 'Model Accuracy', 
      value: modelStatus.isTrained ? `${(modelStatus.accuracy * 100).toFixed(1)}%` : 'N/A', 
      icon: Activity, 
      color: 'text-green-400', 
      bg: 'bg-green-400/10' 
    },
  ];

  const chartData = faces.map(f => ({
    name: f.name,
    samples: f.sampleCount
  })).sort((a, b) => b.samples - a.samples).slice(0, 10);

  if (isLoading) {
    return (
      <div className="h-full flex flex-col items-center justify-center space-y-4">
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-gray-400">Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white mb-2">System Overview</h1>
        <p className="text-gray-400">Real-time metrics of the face recognition database.</p>
      </div>

      {!modelStatus.isTrained && faces.length > 0 && (
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4 flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-amber-500 mt-0.5 flex-shrink-0" />
          <div>
            <h3 className="text-amber-400 font-medium">Model Out of Sync</h3>
            <p className="text-amber-200/60 text-sm mt-1">
              New face data has been added but the model hasn't been retrained. 
              Go to the Training section to update the model.
            </p>
          </div>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map((stat, idx) => (
          <div key={idx} className="bg-gray-900 border border-gray-800 p-6 rounded-xl hover:border-gray-700 transition-colors">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-500 text-sm font-medium uppercase tracking-wider">{stat.label}</p>
                <p className="text-3xl font-bold text-white mt-2">{stat.value}</p>
              </div>
              <div className={`p-3 rounded-lg ${stat.bg}`}>
                <stat.icon className={`w-6 h-6 ${stat.color}`} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Chart Section */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-6">Sample Distribution (Top 10)</h2>
        <div className="h-80 w-full">
          {faces.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                <XAxis 
                  dataKey="name" 
                  stroke="#9ca3af" 
                  tick={{ fill: '#9ca3af' }} 
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis 
                  stroke="#9ca3af" 
                  tick={{ fill: '#9ca3af' }} 
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#fff' }}
                  itemStyle={{ color: '#fff' }}
                  cursor={{ fill: 'rgba(55, 65, 81, 0.4)' }}
                />
                <Bar dataKey="samples" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill="#3b82f6" fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-gray-500 space-y-2">
              <Database className="w-12 h-12 opacity-20" />
              <p>No data available to display</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
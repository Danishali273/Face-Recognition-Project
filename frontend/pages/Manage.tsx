import React, { useEffect, useState } from 'react';
import { Trash2, Search, User, Calendar, Database, RefreshCw } from 'lucide-react';
import { getStoredFaces, deleteFace, resetModelStatus } from '../services/faceService';
import { FaceData } from '../types';

const Manage: React.FC = () => {
  const [faces, setFaces] = useState<FaceData[]>([]);
  const [search, setSearch] = useState('');
  const [isDeleting, setIsDeleting] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getStoredFaces();
      setFaces(data);
    } catch (err) {
      setError('Failed to load faces. Make sure the API server is running.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (name: string) => {
    if (!window.confirm(`Are you sure you want to delete "${name}"? This action cannot be undone.`)) return;
    
    setIsDeleting(name);
    try {
      await deleteFace(name);
      resetModelStatus();
      await loadData();
    } catch (err) {
      setError('Failed to delete face.');
      console.error(err);
    } finally {
      setIsDeleting(null);
    }
  };

  const filteredFaces = faces.filter(f => 
    f.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">Manage Data</h1>
          <p className="text-gray-400">View and manage collected training samples.</p>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={loadData}
            disabled={isLoading}
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
            title="Refresh"
          >
            <RefreshCw className={`w-5 h-5 text-gray-400 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
            <input 
              type="text" 
              placeholder="Search names..." 
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full md:w-64 bg-gray-900 border border-gray-700 text-white pl-10 pr-4 py-2 rounded-lg focus:ring-2 focus:ring-green-500 focus:outline-none"
            />
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-gray-950/50 border-b border-gray-800 text-gray-400 text-sm uppercase tracking-wider">
                <th className="px-6 py-4 font-medium">Identity</th>
                <th className="px-6 py-4 font-medium">Samples</th>
                <th className="px-6 py-4 font-medium">Last Updated</th>
                <th className="px-6 py-4 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {isLoading ? (
                <tr>
                  <td colSpan={4} className="px-6 py-12 text-center text-gray-500">
                    <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                    Loading...
                  </td>
                </tr>
              ) : filteredFaces.length > 0 ? (
                filteredFaces.map((face) => (
                  <tr key={face.id} className="hover:bg-gray-800/50 transition-colors group">
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400">
                          <User className="w-5 h-5" />
                        </div>
                        <span className="font-medium text-white">{face.name}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2 text-gray-300">
                        <Database className="w-4 h-4 text-gray-500" />
                        <span>{face.sampleCount}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-gray-400">
                      <div className="flex items-center space-x-2">
                        <Calendar className="w-4 h-4 text-gray-500" />
                        <span>{new Date(face.lastUpdated).toLocaleDateString()}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button 
                        onClick={() => handleDelete(face.name)}
                        disabled={isDeleting === face.name}
                        className="p-2 text-gray-500 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-colors"
                        title="Delete Face Data"
                      >
                        {isDeleting === face.name ? (
                          <div className="w-5 h-5 border-2 border-red-500 border-t-transparent rounded-full animate-spin" />
                        ) : (
                          <Trash2 className="w-5 h-5" />
                        )}
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} className="px-6 py-12 text-center text-gray-500">
                    {search ? 'No matches found.' : 'No face data available. Capture some faces first.'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Manage;
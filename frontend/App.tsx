import React from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Capture from './pages/Capture';
import Manage from './pages/Manage';
import Train from './pages/Train';
import Recognize from './pages/Recognize';

const App: React.FC = () => {
  return (
    <HashRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/capture" element={<Capture />} />
          <Route path="/manage" element={<Manage />} />
          <Route path="/train" element={<Train />} />
          <Route path="/recognize" element={<Recognize />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </HashRouter>
  );
};

export default App;
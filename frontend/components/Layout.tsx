import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Camera, 
  Users, 
  BrainCircuit, 
  ScanFace,
  Menu,
  X
} from 'lucide-react';
import { AppRoute } from '../types';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [isSidebarOpen, setIsSidebarOpen] = React.useState(false);
  const location = useLocation();

  const navItems = [
    { name: 'Dashboard', icon: LayoutDashboard, path: AppRoute.DASHBOARD },
    { name: 'Capture Faces', icon: Camera, path: AppRoute.CAPTURE },
    { name: 'Manage Data', icon: Users, path: AppRoute.MANAGE },
    { name: 'Train Model', icon: BrainCircuit, path: AppRoute.TRAIN },
    { name: 'Live Recognition', icon: ScanFace, path: AppRoute.RECOGNIZE },
  ];

  return (
    <div className="flex h-screen bg-gray-950 text-white overflow-hidden font-sans">
      {/* Mobile Sidebar Overlay */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-20 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        fixed inset-y-0 left-0 z-30 w-64 bg-gray-900 border-r border-gray-800 transform transition-transform duration-200 ease-in-out
        lg:translate-x-0 lg:static lg:inset-auto
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="flex items-center justify-between h-16 px-6 border-b border-gray-800">
          <div className="flex items-center space-x-2">
            <ScanFace className="w-6 h-6 text-green-500" />
            <span className="text-xl font-bold tracking-tight">FaceGuard</span>
          </div>
          <button 
            onClick={() => setIsSidebarOpen(false)}
            className="lg:hidden text-gray-400 hover:text-white"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <nav className="p-4 space-y-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => setIsSidebarOpen(false)}
                className={`
                  flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors
                  ${isActive 
                    ? 'bg-green-500/10 text-green-400 border border-green-500/20' 
                    : 'text-gray-400 hover:bg-gray-800 hover:text-white'}
                `}
              >
                <item.icon className={`w-5 h-5 mr-3 ${isActive ? 'text-green-400' : 'text-gray-500'}`} />
                {item.name}
              </NavLink>
            );
          })}
        </nav>

        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-800">
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs text-gray-500 uppercase tracking-wider font-semibold">System Online</span>
          </div>
          <div className="mt-2 text-xs text-gray-600">
            v1.0.0 • Local Environment
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        {/* Mobile Header */}
        <header className="flex items-center justify-between h-16 px-4 border-b border-gray-800 bg-gray-900 lg:hidden flex-shrink-0">
          <div className="flex items-center space-x-2">
             <ScanFace className="w-6 h-6 text-green-500" />
             <span className="font-bold">FaceGuard</span>
          </div>
          <button 
            onClick={() => setIsSidebarOpen(true)}
            className="p-2 text-gray-400 hover:text-white rounded-md hover:bg-gray-800"
          >
            <Menu className="w-6 h-6" />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-4 lg:p-8 scroll-smooth">
          <div className="max-w-7xl mx-auto h-full">
            {children}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Layout;
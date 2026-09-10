import React from 'react';
import { UrbanEyeProvider, useUrbanEye } from './context/UrbanEyeContext';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { NotificationToastContainer } from './components/notifications/NotificationToast';
import { Dashboard } from './pages/Dashboard';
import { EdgeAI } from './pages/EdgeAI';
import { EventEngine } from './pages/EventEngine';
import { Fleet } from './pages/Fleet';
import { EventLog } from './pages/EventLog';
import { Infrastructure } from './pages/Infrastructure';
import { TrafficMobility } from './pages/TrafficMobility';
import { SafetyIncidents } from './pages/SafetyIncidents';

const MainContent: React.FC = () => {
  const { currentModule } = useUrbanEye();

  const renderModule = () => {
    switch (currentModule) {
      case 'edge-ai':
        return <EdgeAI />;
      case 'event-engine':
        return <EventEngine />;
      case 'dashboard':
        return <Dashboard />;
      case 'fleet':
        return <Fleet />;
      case 'event-log':
        return <EventLog />;
      case 'infrastructure':
        return <Infrastructure />;
      case 'traffic':
        return <TrafficMobility />;
      case 'safety':
        return <SafetyIncidents />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
      <Header />
      <main className="flex-1 pb-12">
        {renderModule()}
      </main>
      <footer className="px-8 py-4 border-t border-[#46707E]/15 bg-white/40 text-xs text-gray-500 flex flex-col sm:flex-row items-center justify-between gap-2">
        <div className="flex items-center space-x-2">
          <span className="font-bold text-[#22393C]">UrbanEye</span>
          <span>•</span>
          <span>PS 26124: Mobile Urban Intelligence Platform</span>
        </div>
        <div className="text-[11px] font-mono text-[#46707E]">
          SIH 2026 Prototype • Edge AI Video Processing & Central Intelligence
        </div>
      </footer>
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <UrbanEyeProvider>
      <div className="flex min-h-screen bg-[#F6F7F3] relative selection:bg-[#AFBB98]/40 selection:text-[#22393C]">
        <Sidebar />
        <MainContent />
        <NotificationToastContainer />
      </div>
    </UrbanEyeProvider>
  );
};

export default App;

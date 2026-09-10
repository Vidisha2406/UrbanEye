import React from 'react';
import {
  LayoutDashboard,
  Bus as BusIcon,
  FileText,
  Building2,
  GitFork,
  ShieldAlert,
  Cpu,
  Activity,
  LogOut,
  Settings,
  Radio,
} from 'lucide-react';
import { NavigationModule } from '../../types';
import { useUrbanEye } from '../../context/UrbanEyeContext';

interface NavItem {
  id: NavigationModule;
  label: string;
  moduleNum: string;
  icon: React.ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', moduleNum: '03', icon: LayoutDashboard },
  { id: 'fleet', label: 'Fleet & Buses', moduleNum: '04', icon: BusIcon },
  { id: 'event-log', label: 'Event Log', moduleNum: '05', icon: FileText },
  { id: 'infrastructure', label: 'Infrastructure', moduleNum: '06', icon: Building2 },
  { id: 'traffic', label: 'Traffic & Mobility', moduleNum: '07', icon: GitFork },
  { id: 'safety', label: 'Safety & Incidents', moduleNum: '08', icon: ShieldAlert },
  { id: 'edge-ai', label: 'Edge AI (On Bus)', moduleNum: '01', icon: Cpu },
  { id: 'event-engine', label: 'Event Engine', moduleNum: '02', icon: Activity },
];

export const Sidebar: React.FC = () => {
  const { currentModule, setCurrentModule } = useUrbanEye();

  return (
    <aside className="w-72 min-h-screen bg-[#22393C] text-[#CECDB9] flex flex-col justify-between border-r border-[#46707E]/30 relative z-30 shrink-0 shadow-2xl">
      {/* Top Branding Section */}
      <div>
        <div className="p-6 pb-5 border-b border-[#46707E]/20">
          <div className="flex items-center space-x-3 mb-2">
            {/* Custom UrbanEye Brand SVG Logo */}
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#46707E] via-[#6B8B81] to-[#AFBB98] p-0.5 shadow-lg flex items-center justify-center shrink-0">
              <div className="w-full h-full bg-[#22393C] rounded-[10px] flex items-center justify-center relative overflow-hidden">
                {/* Visual Eye Iris with Bus/Grid Accent */}
                <svg viewBox="0 0 32 32" className="w-6 h-6 text-[#AFBB98]" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 16C3 16 8 7 16 7C24 7 29 16 29 16C29 16 24 25 16 25C8 25 3 16 3 16Z" stroke="#AFBB98" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <circle cx="16" cy="16" r="5" stroke="#CECDB9" strokeWidth="2" fill="#46707E" fillOpacity="0.4" />
                  <circle cx="16" cy="16" r="2" fill="#AFBB98" />
                  <path d="M12 21L20 21" stroke="#AFBB98" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-xl tracking-wider text-white">URBANEYE</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-[#46707E]/40 text-[#AFBB98] border border-[#AFBB98]/30">
                  PS 26124
                </span>
              </div>
              <p className="text-[10px] uppercase tracking-widest text-[#AFBB98] font-medium mt-0.5">
                AI × MOBILITY × SMART CITIES
              </p>
            </div>
          </div>
        </div>

        {/* Navigation List */}
        <nav className="p-3 space-y-1">
          <div className="px-3 pt-2 pb-1.5 text-[10px] uppercase font-bold tracking-widest text-[#AFBB98]/70">
            Platform Modules
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentModule === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setCurrentModule(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group ${
                  isActive
                    ? 'bg-gradient-to-r from-[#46707E]/60 to-[#6B8B81]/40 text-white shadow-md border border-[#AFBB98]/40 translate-x-1'
                    : 'text-[#CECDB9]/80 hover:text-white hover:bg-white/5'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <div
                    className={`p-1.5 rounded-lg transition-colors ${
                      isActive ? 'bg-[#AFBB98]/20 text-[#AFBB98]' : 'text-[#6B8B81] group-hover:text-[#AFBB98]'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="tracking-wide text-left">{item.label}</span>
                </div>
                <span
                  className={`text-[10px] font-mono px-1.5 py-0.5 rounded transition-colors ${
                    isActive ? 'bg-[#AFBB98] text-[#22393C] font-bold' : 'text-[#6B8B81] group-hover:text-[#CECDB9]'
                  }`}
                >
                  {item.moduleNum}
                </span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Profile / Settings Area */}
      <div className="p-4 border-t border-[#46707E]/20 bg-black/15">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-full bg-[#46707E]/40 border border-[#AFBB98]/30 flex items-center justify-center text-xs font-bold text-[#AFBB98]">
              PMC
            </div>
            <div>
              <div className="text-xs font-semibold text-white">Pune Smart City</div>
              <div className="text-[10px] text-[#AFBB98] flex items-center space-x-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block"></span>
                <span>Command Center Live</span>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-1 text-[#CECDB9]/60">
            <button
              onClick={() => alert('UrbanEye System Configuration v1.0.0')}
              className="p-1.5 rounded-lg hover:text-white hover:bg-white/10 transition-colors"
              title="Settings"
            >
              <Settings className="w-4 h-4" />
            </button>
            <button
              onClick={() => alert('Smart City Session Active')}
              className="p-1.5 rounded-lg hover:text-white hover:bg-white/10 transition-colors"
              title="Log Out"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </aside>
  );
};

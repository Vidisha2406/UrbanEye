import React from 'react';
import { CheckCircle2, X } from 'lucide-react';
import { useUrbanEye } from '../../context/UrbanEyeContext';

export const NotificationToastContainer: React.FC = () => {
  const { notifications, dismissNotification } = useUrbanEye();

  if (notifications.length === 0) return null;

  return (
    <div className="fixed top-6 right-6 z-50 flex flex-col space-y-3 pointer-events-none max-w-sm w-full">
      {notifications.map((n) => (
        <div
          key={n.id}
          className="pointer-events-auto bg-[#22393C]/95 text-white backdrop-blur-xl border border-[#AFBB98]/40 rounded-xl p-4 shadow-2xl transition-all duration-300 transform translate-y-0 animate-in fade-in slide-in-from-top-4"
        >
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3">
              <div className="p-1 rounded-full bg-[#AFBB98]/20 text-[#AFBB98] mt-0.5 shrink-0">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <h4 className="text-sm font-bold text-white tracking-wide flex items-center gap-1.5">
                  ✓ {n.title}
                </h4>
                <p className="text-xs text-[#CECDB9] mt-1 leading-relaxed">
                  {n.message}
                </p>
                <div className="flex items-center space-x-2 mt-2">
                  <span className="text-[10px] uppercase tracking-wider font-mono text-[#AFBB98]">
                    {n.timestamp}
                  </span>
                  <span className="text-[10px] text-white/40">•</span>
                  <span className="text-[10px] text-[#AFBB98] font-medium">
                    Central Telemetry Synced
                  </span>
                </div>
              </div>
            </div>
            <button
              onClick={() => dismissNotification(n.id)}
              className="text-[#CECDB9]/60 hover:text-white transition-colors p-1 -mr-1 -mt-1 rounded-md"
              title="Dismiss"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

import React, { useState } from 'react';
import {
  GitFork,
  Clock,
  TrendingUp,
  AlertCircle,
  Eye,
  MapPin,
  ArrowRight,
  Gauge,
  Car,
} from 'lucide-react';
import { TrafficCongestionMap } from '../components/maps/TrafficCongestionMap';
import { useUrbanEye } from '../context/UrbanEyeContext';
import { TrafficItem } from '../types';

export const TrafficMobility: React.FC = () => {
  const { traffic, latestTrafficTelemetry } = useUrbanEye();
  const [selectedItem, setSelectedItem] = useState<TrafficItem | null>(null);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* Top Banner */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center space-x-4">
          <div className="p-3 rounded-xl bg-[#22393C] text-[#AFBB98] shadow-md">
            <GitFork className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-[#22393C]">
              Traffic & Mobility Intelligence
            </h2>
            <p className="text-xs text-[#46707E] font-medium mt-0.5">
              Automated Bus Fleet Telemetry: Bottleneck Detection, Corridor Transit Times & Route Delay Estimation
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4 text-xs font-mono">
          <div className="px-3 py-1.5 rounded-xl bg-white/80 border border-[#46707E]/20 text-[#22393C]">
            Avg City Corridor Speed: <strong>14.2 km/hr</strong>
          </div>
          <div className="px-3 py-1.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 font-bold">
            Peak Delay Index: <strong>+18.4%</strong>
          </div>
        </div>
      </div>

      {/* Compact Vehicle Flow & Density Area */}
      <div className="glass-panel p-4 lg:p-5 rounded-2xl border border-[#46707E]/20">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-[#22393C] text-[#AFBB98] shadow-sm shrink-0">
              <Car className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="text-[10px] font-extrabold uppercase tracking-wider text-[#46707E]">
                  Edge Video Classification & Counting
                </span>
              </div>
              <h3 className="text-sm font-extrabold text-[#22393C]">
                Vehicle Flow
              </h3>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="px-3.5 py-1.5 rounded-xl bg-white/80 border border-[#46707E]/20">
              <span className="text-[10px] font-bold uppercase text-gray-500 block">
                Current Vehicle Count
              </span>
              <span className="text-sm font-mono font-black text-[#22393C]">
                {latestTrafficTelemetry ? `${latestTrafficTelemetry.vehicleCount} vehicles` : '68 vehicles/min'}
              </span>
            </div>

            <div className="px-3.5 py-1.5 rounded-xl bg-rose-50 border border-rose-200">
              <span className="text-[10px] font-bold uppercase text-rose-700 block">
                Traffic Density
              </span>
              <span className="text-sm font-mono font-black text-rose-800">
                {latestTrafficTelemetry ? latestTrafficTelemetry.trafficDensity : 'HIGH'}
              </span>
            </div>

            <div className="px-3.5 py-1.5 rounded-xl bg-amber-50 border border-amber-200">
              <span className="text-[10px] font-bold uppercase text-amber-700 block">
                Congestion Level
              </span>
              <span className="text-sm font-mono font-black text-amber-800">
                {latestTrafficTelemetry ? latestTrafficTelemetry.congestionLevel : 'HIGH'}
              </span>
            </div>

            <div className="px-3.5 py-1.5 rounded-xl bg-white/80 border border-[#46707E]/20">
              <span className="text-[10px] font-bold uppercase text-gray-500 block mb-0.5">
                Vehicle Mix
              </span>
              <div className="flex flex-wrap items-center gap-2 text-xs font-mono font-bold">
                <span className="text-[#22393C]">Cars: {latestTrafficTelemetry?.vehicleMix?.Car || '54%'}</span>
                <span className="text-gray-300">•</span>
                <span className="text-[#46707E]">Two-wheelers: {latestTrafficTelemetry?.vehicleMix?.Motorcycle || '31%'}</span>
                <span className="text-gray-300">•</span>
                <span className="text-[#6B8B81]">Buses: {latestTrafficTelemetry?.vehicleMix?.Bus || '9%'}</span>
                <span className="text-gray-300">•</span>
                <span className="text-amber-800">Trucks: {latestTrafficTelemetry?.vehicleMix?.Truck || '6%'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Traffic Congestion Map (STRICTLY TRAFFIC DATA ONLY) */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-extrabold text-[#22393C] uppercase tracking-wide">
              Dedicated Traffic Congestion Heatmap
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Displaying exclusively traffic delay corridors, bottlenecks, and speed persistence
            </p>
          </div>
        </div>

        <TrafficCongestionMap />
      </div>

      {/* ONE Consolidated Traffic / Mobility Table */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-extrabold text-[#22393C] uppercase tracking-wide">
            Consolidated Traffic & Mobility Table
          </h3>
          <span className="text-xs text-gray-500 font-mono">
            {traffic.length} Monitored Corridors
          </span>
        </div>

        <div className="overflow-x-auto rounded-xl border border-[#46707E]/20">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#22393C]/5 text-[#46707E] font-bold uppercase tracking-wider border-b border-[#46707E]/15">
              <tr>
                <th className="py-3.5 px-4">Location / Route</th>
                <th className="py-3.5 px-4">Type</th>
                <th className="py-3.5 px-4">Current Status</th>
                <th className="py-3.5 px-4">Severity</th>
                <th className="py-3.5 px-4">Avg. Delay</th>
                <th className="py-3.5 px-4">Persistence</th>
                <th className="py-3.5 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#46707E]/10 bg-white/40">
              {traffic.map((item) => (
                <tr
                  key={item.id}
                  className="hover:bg-white/80 transition-colors cursor-pointer"
                  onClick={() => setSelectedItem(item)}
                >
                  <td className="py-3.5 px-4 font-bold text-[#22393C]">
                    <div className="flex items-center space-x-2">
                      <MapPin className="w-3.5 h-3.5 text-[#46707E]" />
                      <span>{item.location}</span>
                    </div>
                  </td>
                  <td className="py-3.5 px-4">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        item.type === 'Bottleneck'
                          ? 'bg-rose-100 text-rose-800'
                          : 'bg-amber-100 text-amber-800'
                      }`}
                    >
                      {item.type}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 font-medium text-[#46707E]">
                    {item.currentStatus}
                  </td>
                  <td className="py-3.5 px-4">
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        item.severity === 'High'
                          ? 'bg-rose-100 text-rose-700'
                          : 'bg-amber-100 text-amber-700'
                      }`}
                    >
                      {item.severity}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 font-mono font-bold text-rose-700">
                    {item.avgDelay}
                  </td>
                  <td className="py-3.5 px-4 font-mono text-gray-600">
                    {item.persistence}
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedItem(item);
                      }}
                      className="px-3 py-1 rounded-lg text-xs font-bold bg-[#46707E]/10 hover:bg-[#46707E]/20 text-[#46707E] transition-all"
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Corridor Inspection Modal / Drawer */}
      {selectedItem && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white/95 backdrop-blur-xl border border-[#46707E]/30 rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-3">
              <div>
                <h3 className="text-base font-bold text-[#22393C]">{selectedItem.location}</h3>
                <span className="text-xs text-[#46707E] font-medium">{selectedItem.type} Corridor</span>
              </div>
              <span
                className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${
                  selectedItem.severity === 'High'
                    ? 'bg-rose-100 text-rose-700'
                    : 'bg-amber-100 text-amber-700'
                }`}
              >
                {selectedItem.severity} Severity
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-gray-50">
                <span className="text-gray-500">Average Delay</span>
                <div className="text-lg font-mono font-bold text-rose-600 mt-0.5">
                  {selectedItem.avgDelay}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-gray-50">
                <span className="text-gray-500">Queue Persistence</span>
                <div className="text-lg font-mono font-bold text-[#22393C] mt-0.5">
                  {selectedItem.persistence}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-gray-50">
                <span className="text-gray-500">Transit Speed</span>
                <div className="text-lg font-mono font-bold text-[#46707E] mt-0.5">
                  {selectedItem.speedKmh} km/hr
                </div>
              </div>
              <div className="p-3 rounded-xl bg-gray-50">
                <span className="text-gray-500">GPS Coordinates</span>
                <div className="text-xs font-mono font-bold text-gray-700 mt-1">
                  {selectedItem.coordinates[0]}, {selectedItem.coordinates[1]}
                </div>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-[#22393C]/5 text-xs text-[#22393C]">
              <strong>Traffic Intelligence Recommendation:</strong> Signal phase cycle adjustment recommended at upstream junction to relieve tailback queue.
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setSelectedItem(null)}
                className="px-4 py-2 rounded-xl bg-[#22393C] text-white text-xs font-bold hover:bg-[#2C494D]"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

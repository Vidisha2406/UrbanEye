import React, { useState, useMemo } from 'react';
import {
  Bus as BusIcon,
  Search,
  Filter,
  CheckCircle2,
  XCircle,
  Eye,
  MapPin,
  Clock,
  Gauge,
  Navigation,
  Layers,
  AlertCircle,
  X,
} from 'lucide-react';
import { LightCityMap } from '../components/maps/LightCityMap';
import { useUrbanEye } from '../context/UrbanEyeContext';
import { UrbanEvent } from '../types';

export const Fleet: React.FC = () => {
  const { buses, selectedBusId, setSelectedBusId, events, setCurrentModule } = useUrbanEye();
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'Online' | 'Offline'>('ALL');
  const [routeFilter, setRouteFilter] = useState<string>('ALL');
  const [inspectedIssue, setInspectedIssue] = useState<UrbanEvent | null>(null);

  const selectedBus = buses.find((b) => b.id === selectedBusId) || buses[0];

  // Complete issues associated with the selected bus
  const busIssues = useMemo(() => {
    const matched = events.filter((e) => e.busId === selectedBus.id);
    if (matched.length > 0) return matched;
    // If bus has open issues but none in event store, create realistic ones
    if (selectedBus.openIssues > 0) {
      return Array.from({ length: selectedBus.openIssues }).map((_, i) => ({
        id: `EVT-${selectedBus.id.replace('BUS-', '')}0${i + 1}`,
        type: i % 2 === 0 ? 'ROAD_DAMAGE' : 'TRAFFIC',
        category: (i % 2 === 0 ? 'Pothole' : 'Traffic') as any,
        className: i % 2 === 0 ? 'Pothole Surface Cavity' : 'Corridor Slowdown',
        confidence: 90 + i * 2,
        severity: (i === 0 ? 'HIGH' : i === 1 ? 'MEDIUM' : 'LOW') as any,
        busId: selectedBus.id,
        timestamp: '2026-04-22 10:15:00',
        timeFormatted: selectedBus.lastUpdated,
        locationName: `${selectedBus.location}, Pune`,
        coordinates: selectedBus.coordinates,
        evidenceFrame: `frame_bus_${selectedBus.id.toLowerCase()}_${i + 1}.jpg`,
        status: (i === 0 ? 'New' : i === 1 ? 'Confirmed' : 'In Progress') as any,
        contextNote: `Edge detection captured by ${selectedBus.id} onboard camera near ${selectedBus.location}.`,
        isTransmitted: true,
      }));
    }
    return [];
  }, [events, selectedBus]);

  // Unique routes for filter dropdown
  const uniqueRoutes = useMemo(() => {
    return Array.from(new Set(buses.map((b) => b.route)));
  }, [buses]);

  // Filtered bus list
  const filteredBuses = useMemo(() => {
    return buses.filter((b) => {
      const matchSearch =
        b.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        b.location.toLowerCase().includes(searchTerm.toLowerCase()) ||
        b.driverId.toLowerCase().includes(searchTerm.toLowerCase()) ||
        b.route.toLowerCase().includes(searchTerm.toLowerCase());
      const matchStatus = statusFilter === 'ALL' || b.status === statusFilter;
      const matchRoute = routeFilter === 'ALL' || b.route === routeFilter;
      return matchSearch && matchStatus && matchRoute;
    });
  }, [buses, searchTerm, statusFilter, routeFilter]);

  const activeCount = buses.filter((b) => b.status === 'Online').length;
  const inactiveCount = buses.filter((b) => b.status === 'Offline').length;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* 4 Top KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="glass-panel p-5 rounded-2xl">
          <span className="text-xs font-bold uppercase tracking-wider text-[#46707E]">
            Total Buses
          </span>
          <div className="text-3xl font-extrabold text-[#22393C] mt-2 font-mono">
            {buses.length}
          </div>
          <p className="text-xs text-gray-500 mt-1">PMPML Edge-Equipped Fleet</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl">
          <span className="text-xs font-bold uppercase tracking-wider text-emerald-800">
            Active
          </span>
          <div className="text-3xl font-extrabold text-emerald-700 mt-2 font-mono">
            {activeCount}
          </div>
          <p className="text-xs text-emerald-600 mt-1">Streaming Video & Telemetry</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl">
          <span className="text-xs font-bold uppercase tracking-wider text-gray-600">
            Inactive
          </span>
          <div className="text-3xl font-extrabold text-gray-700 mt-2 font-mono">
            {inactiveCount}
          </div>
          <p className="text-xs text-gray-500 mt-1">In Depot / Maintenance</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl">
          <span className="text-xs font-bold uppercase tracking-wider text-[#46707E]">
            On Route
          </span>
          <div className="text-3xl font-extrabold text-[#22393C] mt-2 font-mono">
            16
          </div>
          <p className="text-xs text-[#46707E] font-medium mt-1">Active Passenger Transit</p>
        </div>
      </div>

      {/* Selected Bus Details Card (Embedded in View - NO separate page) */}
      {selectedBus && (
        <div id="bus-details-panel" className="glass-panel p-6 rounded-2xl border-2 border-[#46707E]/30 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#46707E]/20 pb-4">
            <div className="flex items-center space-x-3">
              <div className="p-3 rounded-xl bg-[#22393C] text-[#AFBB98] shadow">
                <BusIcon className="w-6 h-6" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <h3 className="text-xl font-extrabold text-[#22393C]">{selectedBus.id}</h3>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-[#46707E]/15 text-[#46707E]">
                    {selectedBus.route}
                  </span>
                  <span
                    className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${
                      selectedBus.status === 'Online'
                        ? 'bg-emerald-100 text-emerald-800'
                        : 'bg-gray-200 text-gray-700'
                    }`}
                  >
                    {selectedBus.status}
                  </span>
                </div>
                <p className="text-xs text-[#46707E] mt-0.5">
                  Driver: <strong>{selectedBus.driverId}</strong> • Last Updated:{' '}
                  <strong>{selectedBus.lastUpdated}</strong>
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={() => setCurrentModule('edge-ai')}
                className="px-4 py-2 rounded-xl bg-[#22393C] text-[#CECDB9] text-xs font-bold hover:bg-[#2C494D] transition-all flex items-center space-x-2 shadow-sm"
              >
                <span>Switch to Onboard Camera</span>
                <span>→</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4 text-xs">
            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-gray-500 font-medium">Current Speed</span>
              <div className="text-base font-mono font-bold text-[#22393C] mt-1">
                {selectedBus.speed}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-gray-500 font-medium">Location</span>
              <div className="text-sm font-semibold text-[#22393C] mt-1 truncate">
                {selectedBus.location}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-gray-500 font-medium">Today's Distance</span>
              <div className="text-base font-mono font-bold text-[#46707E] mt-1">
                {selectedBus.distanceToday}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-gray-500 font-medium">Total Events</span>
              <div className="text-base font-mono font-bold text-[#22393C] mt-1">
                {selectedBus.totalEvents}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-gray-500 font-medium">Open Issues</span>
              <div className="text-base font-mono font-bold text-amber-700 mt-1">
                {selectedBus.openIssues}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-gray-500 font-medium">GPS Telemetry</span>
              <div className="text-[11px] font-mono font-bold text-[#46707E] mt-1 truncate">
                {selectedBus.coordinates[0].toFixed(4)}, {selectedBus.coordinates[1].toFixed(4)}
              </div>
            </div>
          </div>

          {/* Associated Issues & Events for this Bus */}
          <div className="pt-4 border-t border-[#46707E]/20 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-amber-700" />
                <h4 className="text-xs font-extrabold uppercase tracking-wider text-[#22393C]">
                  Issues & Detections Associated with {selectedBus.id} ({busIssues.length})
                </h4>
              </div>
              <span className="text-[11px] text-gray-500 font-mono">
                Real-Time Edge Telemetry Feed
              </span>
            </div>

            {busIssues.length === 0 ? (
              <div className="p-4 rounded-xl bg-white/60 text-xs text-gray-500 text-center font-medium">
                No active issues reported for {selectedBus.id}. Telemetry status is nominal.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {busIssues.map((issue, idx) => (
                  <div
                    key={issue.id}
                    className="p-3.5 rounded-xl bg-white/85 border border-[#46707E]/20 hover:border-[#46707E]/40 hover:bg-white transition-all space-y-2 shadow-sm"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-bold text-[#46707E] font-mono">
                          {idx + 1}.
                        </span>
                        <h5 className="font-bold text-xs text-[#22393C]">{issue.className}</h5>
                      </div>
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          issue.severity === 'HIGH' || issue.severity === 'CRITICAL'
                            ? 'bg-rose-100 text-rose-700 border border-rose-200'
                            : issue.severity === 'MEDIUM'
                            ? 'bg-amber-100 text-amber-800'
                            : 'bg-emerald-100 text-emerald-800'
                        }`}
                      >
                        {issue.severity}
                      </span>
                    </div>

                    <div className="space-y-1 text-xs text-[#22393C]/80">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Location:</span>
                        <span className="font-semibold text-[#22393C] truncate">{issue.locationName}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Status:</span>
                        <span
                          className={`px-1.5 py-0.2 rounded font-bold text-[10px] ${
                            issue.status === 'New'
                              ? 'bg-blue-100 text-blue-800'
                              : issue.status === 'Confirmed'
                              ? 'bg-amber-100 text-amber-800'
                              : issue.status === 'In Progress'
                              ? 'bg-purple-100 text-purple-800'
                              : 'bg-emerald-100 text-emerald-800'
                          }`}
                        >
                          {issue.status}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Detected:</span>
                        <span className="font-mono text-gray-700">{issue.timeFormatted}</span>
                      </div>
                    </div>

                    {issue.contextNote && (
                      <p className="text-[11px] text-gray-500 line-clamp-1 italic">
                        "{issue.contextNote}"
                      </p>
                    )}

                    <div className="pt-2 border-t border-[#46707E]/10 flex items-center justify-between">
                      <span className="text-[10px] font-mono text-gray-400">
                        {issue.id}
                      </span>
                      <button
                        onClick={() => setInspectedIssue(issue)}
                        className="text-[11px] font-bold text-[#46707E] hover:text-[#22393C] flex items-center gap-1 hover:underline"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span>Inspect Evidence</span>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Fleet Filter Bar */}
      <div className="glass-panel p-4 rounded-xl flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3 flex-1 min-w-[280px]">
          {/* Search Input */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by Bus ID, Route, Driver, Location..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 rounded-xl bg-white border border-[#46707E]/20 text-xs text-[#22393C] focus:outline-none focus:ring-2 focus:ring-[#46707E]"
            />
          </div>

          {/* Status Filter */}
          <div className="flex items-center space-x-1 bg-white p-1 rounded-xl border border-[#46707E]/20 text-xs">
            {(['ALL', 'Online', 'Offline'] as const).map((s) => (
              <button
                key={s}
                onClick={() => setStatusFilter(s)}
                className={`px-3 py-1 rounded-lg font-medium transition-all ${
                  statusFilter === s
                    ? 'bg-[#22393C] text-white font-semibold'
                    : 'text-[#46707E] hover:text-[#22393C]'
                }`}
              >
                {s === 'ALL' ? 'All Status' : s}
              </button>
            ))}
          </div>

          {/* Route Filter */}
          <select
            value={routeFilter}
            onChange={(e) => setRouteFilter(e.target.value)}
            className="bg-white border border-[#46707E]/20 text-xs text-[#22393C] font-medium rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#46707E]"
          >
            <option value="ALL">All Routes</option>
            {uniqueRoutes.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>

        <div className="text-xs text-gray-500 font-mono font-medium">
          Showing <strong>{filteredBuses.length}</strong> of {buses.length} buses
        </div>
      </div>

      {/* Complete Bus Table */}
      <div className="glass-panel rounded-2xl overflow-hidden shadow-sm border border-[#46707E]/20">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#22393C]/5 text-[#46707E] font-bold uppercase tracking-wider border-b border-[#46707E]/15">
              <tr>
                <th className="py-3.5 px-4">Bus ID</th>
                <th className="py-3.5 px-4">Route</th>
                <th className="py-3.5 px-4">Status</th>
                <th className="py-3.5 px-4">Driver ID</th>
                <th className="py-3.5 px-4">Speed</th>
                <th className="py-3.5 px-4">Last Updated</th>
                <th className="py-3.5 px-4">Location</th>
                <th className="py-3.5 px-4 text-center">Open Issues</th>
                <th className="py-3.5 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#46707E]/10">
              {filteredBuses.map((bus) => {
                const isSelected = bus.id === selectedBusId;
                return (
                  <tr
                    key={bus.id}
                    onClick={() => setSelectedBusId(bus.id)}
                    className={`hover:bg-white/80 transition-colors cursor-pointer ${
                      isSelected ? 'bg-[#46707E]/10 font-medium' : 'bg-transparent'
                    }`}
                  >
                    <td className="py-3 px-4 font-mono font-bold text-[#22393C]">
                      <div className="flex items-center space-x-2">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            bus.status === 'Online' ? 'bg-emerald-500' : 'bg-gray-400'
                          }`}
                        ></span>
                        <span>{bus.id}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 font-semibold text-[#46707E]">{bus.route}</td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          bus.status === 'Online'
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'bg-gray-100 text-gray-600'
                        }`}
                      >
                        {bus.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono text-gray-600">{bus.driverId}</td>
                    <td className="py-3 px-4 font-mono font-semibold text-[#22393C]">
                      {bus.speed}
                    </td>
                    <td className="py-3 px-4 text-gray-500">{bus.lastUpdated}</td>
                    <td className="py-3 px-4 font-medium text-[#22393C]">{bus.location}</td>
                    <td className="py-3 px-4 text-center">
                      <span
                        className={`inline-block px-2 py-0.5 rounded-full font-bold font-mono text-xs ${
                          bus.openIssues > 0
                            ? 'bg-amber-100 text-amber-800'
                            : 'bg-gray-100 text-gray-500'
                        }`}
                      >
                        {bus.openIssues}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedBusId(bus.id);
                          document.getElementById('bus-details-panel')?.scrollIntoView({ behavior: 'smooth' });
                        }}
                        className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                          isSelected
                            ? 'bg-[#22393C] text-white shadow-sm'
                            : 'bg-white border border-[#46707E]/20 text-[#46707E] hover:bg-[#46707E]/10'
                        }`}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Embedded Fleet & City Map */}
      <div className="glass-panel p-6 rounded-2xl space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-extrabold text-[#22393C] uppercase tracking-wide">
            Fleet Real-Time Positioning Map
          </h3>
          <span className="text-xs text-[#46707E] font-medium">
            Active GPS Broadcasts from Fleet Corridors
          </span>
        </div>
        <LightCityMap buses={buses} events={events} height="400px" />
      </div>

      {/* Evidence Inspection Modal for Selected Bus Issue */}
      {inspectedIssue && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white/95 backdrop-blur-xl border border-[#46707E]/30 rounded-2xl max-w-xl w-full p-6 space-y-4 shadow-2xl animate-in fade-in zoom-in-95">
            <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-3">
              <div>
                <h3 className="text-base font-bold text-[#22393C]">{inspectedIssue.className}</h3>
                <span className="text-xs font-mono text-[#46707E]">
                  {inspectedIssue.id} • {inspectedIssue.busId} • {inspectedIssue.locationName}
                </span>
              </div>
              <button
                onClick={() => setInspectedIssue(null)}
                className="p-1 rounded-lg text-gray-500 hover:text-black hover:bg-black/5"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Evidence Frame Preview */}
            <div className="relative rounded-xl overflow-hidden bg-black/90 aspect-video flex items-center justify-center border border-[#46707E]/30">
              {(inspectedIssue as any)?.evidenceDataUri || inspectedIssue?.evidenceUrl || (inspectedIssue?.evidenceFrame && (inspectedIssue.evidenceFrame.startsWith('http') || inspectedIssue.evidenceFrame.startsWith('data:'))) ? (
                <div className="relative w-full h-full">
                  <img
                    src={(inspectedIssue as any)?.evidenceDataUri || inspectedIssue?.evidenceUrl || inspectedIssue?.evidenceFrame}
                    alt={`Evidence for ${inspectedIssue.id}`}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      if ((inspectedIssue as any)?.evidenceDataUri && e.currentTarget.src !== (inspectedIssue as any).evidenceDataUri) {
                        e.currentTarget.src = (inspectedIssue as any).evidenceDataUri;
                      } else if (inspectedIssue?.evidenceUrl && e.currentTarget.src !== inspectedIssue.evidenceUrl) {
                        e.currentTarget.src = inspectedIssue.evidenceUrl;
                      }
                    }}
                  />
                  <div className="absolute top-3 right-3 px-2.5 py-1 rounded-md bg-black/75 backdrop-blur-md border border-white/20 text-[10px] font-mono font-bold text-emerald-400 flex items-center space-x-1.5 shadow-md">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span>AI VERIFIED FRAME</span>
                  </div>
                  <div className="absolute bottom-3 left-4 right-4 flex items-center justify-between text-xs font-mono text-white/90 bg-black/75 backdrop-blur-md px-3 py-1 rounded border border-white/10">
                    <span>GPS: {inspectedIssue.coordinates[0].toFixed(4)}, {inspectedIssue.coordinates[1].toFixed(4)}</span>
                    <span>BUS: {inspectedIssue.busId}</span>
                    <span>SEVERITY: {inspectedIssue.severity}</span>
                  </div>
                </div>
              ) : (
                <div className="w-full h-full bg-gradient-to-b from-[#22393C] to-[#121B1C] flex flex-col items-center justify-center p-6 relative">
                  <div className="border-2 border-[#AFBB98] bg-[#AFBB98]/15 rounded p-4 text-center">
                    <span className="text-xs font-mono font-bold text-[#CECDB9]">
                      AI Detection Keyframe: {inspectedIssue.className} ({inspectedIssue.confidence}%)
                    </span>
                    <div className="text-[10px] text-gray-400 mt-1 font-mono">
                      Evidence File: {inspectedIssue.evidenceFrame}
                    </div>
                  </div>

                  <div className="absolute bottom-3 left-4 right-4 flex items-center justify-between text-xs font-mono text-white/80 bg-black/60 px-3 py-1 rounded">
                    <span>GPS: {inspectedIssue.coordinates[0].toFixed(4)}, {inspectedIssue.coordinates[1].toFixed(4)}</span>
                    <span>BUS: {inspectedIssue.busId}</span>
                    <span>SEVERITY: {inspectedIssue.severity}</span>
                  </div>
                </div>
              )}
            </div>

            <div className="p-3 rounded-xl bg-gray-50 border border-gray-200 text-xs text-[#22393C] space-y-1">
              <div className="font-bold text-[#46707E]">AI Sensor Telemetry Context:</div>
              <div>{inspectedIssue.contextNote}</div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setInspectedIssue(null)}
                className="px-4 py-1.5 rounded-xl bg-[#22393C] text-white text-xs font-bold hover:bg-[#2C494D]"
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

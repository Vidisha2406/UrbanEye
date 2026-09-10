import React, { useState, useMemo } from 'react';
import {
  AlertTriangle,
  Search,
  Filter,
  Eye,
  CheckCircle2,
  Clock,
  MapPin,
  Camera,
  Layers,
  X,
  ShieldAlert,
} from 'lucide-react';
import { useUrbanEye } from '../context/UrbanEyeContext';
import { UrbanEvent, IssueStatus, EventCategory } from '../types';

export const EventLog: React.FC = () => {
  const {
    events,
    buses,
    categoryStats,
    selectedEventId,
    setSelectedEventId,
    updateEventStatus,
    corroborateEvent,
  } = useUrbanEye();

  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [selectedBusFilter, setSelectedBusFilter] = useState<string>('ALL');
  const [selectedStatusFilter, setSelectedStatusFilter] = useState<string>('ALL');
  const [dateFilter, setDateFilter] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const inspectedEvent = useMemo(() => {
    if (!selectedEventId) return null;
    return events.find((e) => e.id === selectedEventId) || null;
  }, [events, selectedEventId]);

  const nextSimBus = useMemo(() => {
    if (!inspectedEvent) return 'BUS-05';
    const participated = inspectedEvent.participatingBuses || [inspectedEvent.busId];
    const available = buses.find((b) => !participated.includes(b.id));
    return available ? available.id : 'BUS-05';
  }, [inspectedEvent, buses]);

  // Filter events
  const filteredEvents = useMemo(() => {
    return events.filter((e) => {
      const matchCategory = selectedCategory === 'ALL' || e.category === selectedCategory;
      const matchBus = selectedBusFilter === 'ALL' || e.busId === selectedBusFilter;
      const matchStatus = selectedStatusFilter === 'ALL' || e.status === selectedStatusFilter;
      const matchSearch =
        e.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        e.className.toLowerCase().includes(searchQuery.toLowerCase()) ||
        e.locationName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        e.busId.toLowerCase().includes(searchQuery.toLowerCase());
      return matchCategory && matchBus && matchStatus && matchSearch;
    });
  }, [events, selectedCategory, selectedBusFilter, selectedStatusFilter, searchQuery]);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Pothole':
        return '⚠️';
      case 'Road Damage':
        return '🛠️';
      case 'Traffic':
        return '🚦';
      case 'Vehicle':
        return '🚗';
      case 'Safety / Pedestrian Risk':
        return '🚸';
      case 'Accidents / Incidents':
        return '💥';
      case 'Waterlogging':
        return '🌊';
      case 'Infrastructure Deficiencies':
        return '🏗️';
      case 'ANPR / OCR Events':
        return '📷';
      default:
        return '📍';
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* Top Header Banner */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center space-x-4">
          <div className="p-3 rounded-xl bg-[#22393C] text-[#AFBB98] shadow-md">
            <Layers className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-[#22393C]">Event Log & Issue Management</h2>
            <p className="text-xs text-[#46707E] font-medium mt-0.5">
              Centralized Municipal Audit Trail: Inspect Telemetry Evidence & Track Issue Lifecycle
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono bg-white/70 px-4 py-2 rounded-xl border border-[#46707E]/20">
          <span className="text-gray-500">Registered Events:</span>
          <span className="font-bold text-[#22393C]">{events.length}</span>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="glass-panel p-4 rounded-xl flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3 flex-1 min-w-[280px]">
          {/* Search */}
          <div className="relative flex-1 min-w-[180px]">
            <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search event ID, class, location, bus..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2 rounded-xl bg-white border border-[#46707E]/20 text-xs text-[#22393C] focus:outline-none focus:ring-2 focus:ring-[#46707E]"
            />
          </div>

          {/* All Events / Category Filter */}
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="bg-white border border-[#46707E]/20 text-xs text-[#22393C] font-semibold rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#46707E]"
          >
            <option value="ALL">All Events (All Categories)</option>
            {categoryStats.map((c) => (
              <option key={c.category} value={c.category}>
                {c.category} ({c.count})
              </option>
            ))}
          </select>

          {/* All Buses Filter */}
          <select
            value={selectedBusFilter}
            onChange={(e) => setSelectedBusFilter(e.target.value)}
            className="bg-white border border-[#46707E]/20 text-xs text-[#22393C] font-semibold rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#46707E]"
          >
            <option value="ALL">All Buses</option>
            {buses.map((b) => (
              <option key={b.id} value={b.id}>
                {b.id}
              </option>
            ))}
          </select>

          {/* All Status Filter: New, Confirmed, In Progress, Resolved */}
          <select
            value={selectedStatusFilter}
            onChange={(e) => setSelectedStatusFilter(e.target.value)}
            className="bg-white border border-[#46707E]/20 text-xs text-[#22393C] font-semibold rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#46707E]"
          >
            <option value="ALL">All Statuses</option>
            <option value="New">New</option>
            <option value="Confirmed">Confirmed</option>
            <option value="In Progress">In Progress</option>
            <option value="Resolved">Resolved</option>
          </select>

          {/* Date Filter */}
          <select
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
            className="bg-white border border-[#46707E]/20 text-xs text-[#22393C] font-semibold rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#46707E]"
          >
            <option value="ALL">All Dates (Today)</option>
            <option value="TODAY">Today (2026-04-22)</option>
            <option value="YESTERDAY">Yesterday</option>
            <option value="7DAYS">Past 7 Days</option>
          </select>
        </div>

        <div className="text-xs text-gray-500 font-mono">
          Showing <strong>{filteredEvents.length}</strong> events
        </div>
      </div>

      {/* Events Table */}
      <div className="glass-panel rounded-2xl overflow-hidden shadow-sm border border-[#46707E]/20">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#22393C]/5 text-[#46707E] font-bold uppercase tracking-wider border-b border-[#46707E]/15">
              <tr>
                <th className="py-3.5 px-4">Event ID</th>
                <th className="py-3.5 px-4">Category / Class</th>
                <th className="py-3.5 px-4">Confidence</th>
                <th className="py-3.5 px-4">Severity</th>
                <th className="py-3.5 px-4">Bus ID</th>
                <th className="py-3.5 px-4">Location</th>
                <th className="py-3.5 px-4">Timestamp</th>
                <th className="py-3.5 px-4">Status</th>
                <th className="py-3.5 px-4">Corroboration</th>
                <th className="py-3.5 px-4 text-right">Evidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#46707E]/10">
              {filteredEvents.map((evt) => (
                <tr
                  key={evt.id}
                  onClick={() => setSelectedEventId(evt.id)}
                  className={`hover:bg-white/80 transition-colors cursor-pointer ${
                    selectedEventId === evt.id ? 'bg-[#46707E]/10 font-medium' : 'bg-transparent'
                  }`}
                >
                  <td className="py-3.5 px-4 font-mono font-bold text-[#22393C]">{evt.id}</td>
                  <td className="py-3.5 px-4">
                    <div className="flex items-center space-x-2">
                      <span>{getCategoryIcon(evt.category)}</span>
                      <span className="font-semibold text-[#22393C]">{evt.className}</span>
                    </div>
                  </td>
                  <td className="py-3.5 px-4 font-mono font-bold text-emerald-700">
                    {evt.confidence}%
                  </td>
                  <td className="py-3.5 px-4">
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        evt.severity === 'HIGH' || evt.severity === 'CRITICAL'
                          ? 'bg-rose-100 text-rose-700 border border-rose-200'
                          : 'bg-emerald-100 text-emerald-700'
                      }`}
                    >
                      {evt.severity}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 font-mono font-semibold text-[#46707E]">{evt.busId}</td>
                  <td className="py-3.5 px-4 text-[#22393C]">{evt.locationName}</td>
                  <td className="py-3.5 px-4 text-gray-500 font-mono">{evt.timeFormatted}</td>
                  <td className="py-3.5 px-4">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        evt.status === 'New'
                          ? 'bg-blue-100 text-blue-800'
                          : evt.status === 'Confirmed'
                          ? 'bg-amber-100 text-amber-800'
                          : evt.status === 'In Progress'
                          ? 'bg-purple-100 text-purple-800'
                          : 'bg-emerald-100 text-emerald-800'
                      }`}
                    >
                      {evt.status}
                    </span>
                  </td>
                  <td className="py-3.5 px-4">
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        evt.corroborationStatus === 'VERIFIED'
                          ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                          : evt.corroborationStatus === 'CORROBORATED'
                          ? 'bg-teal-100 text-teal-800 border border-teal-300'
                          : evt.corroborationStatus === 'POTENTIAL_FALSE_POSITIVE'
                          ? 'bg-rose-100 text-rose-800 border border-rose-300'
                          : evt.corroborationStatus === 'NEEDS_VERIFICATION'
                          ? 'bg-amber-100 text-amber-800 border border-amber-300'
                          : 'bg-blue-100 text-blue-800 border border-blue-200'
                      }`}
                    >
                      {evt.corroborationStatus === 'VERIFIED'
                        ? 'Verified (3+ Buses)'
                        : evt.corroborationStatus === 'CORROBORATED'
                        ? `Corroborated (${evt.corroborationCount || 2} Buses)`
                        : evt.corroborationStatus === 'POTENTIAL_FALSE_POSITIVE'
                        ? 'False Positive Alert'
                        : evt.corroborationStatus === 'NEEDS_VERIFICATION'
                        ? 'Needs Verification'
                        : 'Single Bus'}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedEventId(evt.id);
                      }}
                      className="px-3 py-1 rounded-lg text-xs font-bold bg-[#46707E]/10 hover:bg-[#46707E]/20 text-[#46707E] transition-all"
                    >
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detailed Inspection Modal */}
      {inspectedEvent && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white/95 backdrop-blur-xl border border-[#46707E]/30 rounded-2xl max-w-2xl w-full p-6 space-y-5 shadow-2xl animate-in fade-in zoom-in-95">
            <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-4">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">{getCategoryIcon(inspectedEvent.category)}</span>
                <div>
                  <h3 className="text-lg font-bold text-[#22393C]">{inspectedEvent.className}</h3>
                  <span className="text-xs font-mono text-[#46707E] font-semibold">
                    {inspectedEvent.id} • {inspectedEvent.category}
                  </span>
                </div>
              </div>

              <button
                onClick={() => setSelectedEventId(null)}
                className="p-1 rounded-lg text-gray-500 hover:text-black hover:bg-black/5"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Evidence Frame Preview */}
            <div className="relative rounded-xl overflow-hidden bg-black/90 aspect-video flex items-center justify-center border border-[#46707E]/30">
              {(inspectedEvent as any)?.evidenceDataUri || inspectedEvent?.evidenceUrl || (inspectedEvent?.evidenceFrame && (inspectedEvent.evidenceFrame.startsWith('/') || inspectedEvent.evidenceFrame.startsWith('data:') || inspectedEvent.evidenceFrame.startsWith('http'))) ? (
                <div className="relative w-full h-full">
                  <img
                    src={(inspectedEvent as any)?.evidenceDataUri || inspectedEvent?.evidenceUrl || inspectedEvent?.evidenceFrame}
                    alt={`Evidence for ${inspectedEvent.id}`}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      if ((inspectedEvent as any)?.evidenceDataUri && e.currentTarget.src !== (inspectedEvent as any).evidenceDataUri) {
                        e.currentTarget.src = (inspectedEvent as any).evidenceDataUri;
                      } else if (inspectedEvent?.evidenceUrl && e.currentTarget.src !== inspectedEvent.evidenceUrl) {
                        e.currentTarget.src = inspectedEvent.evidenceUrl;
                      }
                    }}
                  />
                  {/* AI Verified Badge */}
                  <div className="absolute top-3 right-3 px-2.5 py-1 rounded-md bg-black/75 backdrop-blur-md border border-white/20 text-[10px] font-mono font-bold text-emerald-400 flex items-center space-x-1.5 shadow-md">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span>AI VERIFIED FRAME</span>
                  </div>

                  {/* Bottom metadata chip */}
                  <div className="absolute bottom-3 left-4 right-4 flex items-center justify-between text-xs font-mono text-white/90 bg-black/75 backdrop-blur-md px-3 py-1 rounded border border-white/10">
                    <span>GPS: {inspectedEvent.coordinates[0]}, {inspectedEvent.coordinates[1]}</span>
                    <span>BUS: {inspectedEvent.busId}</span>
                    <span>TIMESTAMP: {inspectedEvent.timestamp}</span>
                  </div>
                </div>
              ) : (
                <div className="w-full h-full bg-gradient-to-b from-[#22393C] to-[#121B1C] flex flex-col items-center justify-center p-6 relative">
                  <div className="border-2 border-[#AFBB98] bg-[#AFBB98]/15 rounded p-5 text-center">
                    <span className="text-xs font-mono font-bold text-[#CECDB9]">
                      AI Bounding Box: {inspectedEvent.className} ({inspectedEvent.confidence}%)
                    </span>
                    <div className="text-[10px] text-gray-400 mt-1">
                      Telemetry Keyframe: {inspectedEvent.evidenceFrame}
                    </div>
                  </div>

                  <div className="absolute bottom-3 left-4 right-4 flex items-center justify-between text-xs font-mono text-white/80 bg-black/60 px-3 py-1 rounded">
                    <span>GPS: {inspectedEvent.coordinates[0]}, {inspectedEvent.coordinates[1]}</span>
                    <span>BUS: {inspectedEvent.busId}</span>
                    <span>TIMESTAMP: {inspectedEvent.timestamp}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Context & Telemetry Notes */}
            <div className="p-3.5 rounded-xl bg-gray-50 border border-gray-200 text-xs text-[#22393C] space-y-1">
              <div className="font-bold text-[#46707E]">AI Sensor Context:</div>
              <div>{inspectedEvent.contextNote || 'High confidence spatial defect identified.'}</div>
              <div className="text-gray-500 pt-1">
                Location: <strong>{inspectedEvent.locationName}</strong>
              </div>
            </div>

            {/* Multi-Bus Event Corroboration Engine */}
            <div className="p-4 rounded-xl bg-[#22393C]/5 border border-[#46707E]/25 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center space-x-2">
                  <span className="font-bold text-xs text-[#22393C] uppercase tracking-wider">
                    Multi-Bus Corroboration Engine
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                      inspectedEvent.corroborationStatus === 'VERIFIED'
                        ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                        : inspectedEvent.corroborationStatus === 'CORROBORATED'
                        ? 'bg-teal-100 text-teal-800 border border-teal-300'
                        : inspectedEvent.corroborationStatus === 'POTENTIAL_FALSE_POSITIVE'
                        ? 'bg-rose-100 text-rose-800 border border-rose-300'
                        : inspectedEvent.corroborationStatus === 'NEEDS_VERIFICATION'
                        ? 'bg-amber-100 text-amber-800 border border-amber-300'
                        : 'bg-blue-100 text-blue-800 border border-blue-200'
                    }`}
                  >
                    {inspectedEvent.corroborationStatus === 'VERIFIED'
                      ? 'VERIFIED (3+ Fleet Passes)'
                      : inspectedEvent.corroborationStatus === 'CORROBORATED'
                      ? `CORROBORATED (${inspectedEvent.corroborationCount || 2} Buses)`
                      : inspectedEvent.corroborationStatus === 'POTENTIAL_FALSE_POSITIVE'
                      ? 'POTENTIAL FALSE POSITIVE'
                      : inspectedEvent.corroborationStatus === 'NEEDS_VERIFICATION'
                      ? 'NEEDS VERIFICATION (Unconfirmed Pass)'
                      : 'SINGLE BUS INGESTION'}
                  </span>
                </div>

                <div className="text-[11px] font-mono text-[#46707E]">
                  Confirmations: <strong className="text-emerald-700">{inspectedEvent.corroborationCount || 1}</strong> | Non-Confirmations: <strong className="text-rose-700">{inspectedEvent.nonConfirmationCount || 0}</strong>
                </div>
              </div>

              {/* Observation Timeline History */}
              <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
                {(inspectedEvent.observations && inspectedEvent.observations.length > 0
                  ? inspectedEvent.observations
                  : [
                      {
                        id: `OBS-${inspectedEvent.id}-01`,
                        busId: inspectedEvent.busId,
                        route: buses.find((b) => b.id === inspectedEvent.busId)?.route || 'Route 17',
                        timestamp: inspectedEvent.timestamp,
                        observationType: 'CONFIRMED' as const,
                        confidence: inspectedEvent.confidence,
                        notes: `Initial edge AI pass confirmation by ${inspectedEvent.busId}`,
                      },
                    ]
                ).map((obs, idx) => (
                  <div
                    key={obs.id || idx}
                    className="p-2 rounded-lg bg-white border border-[#46707E]/15 flex items-center justify-between text-xs"
                  >
                    <div className="flex items-center space-x-2">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          obs.observationType === 'CONFIRMED' ? 'bg-emerald-500' : 'bg-rose-500'
                        }`}
                      ></span>
                      <span className="font-mono font-bold text-[#22393C]">{obs.busId}</span>
                      <span className="text-[11px] text-gray-500">({obs.route || 'Route 17'})</span>
                      <span className="text-[11px] text-gray-600 truncate max-w-[200px]" title={obs.notes}>
                        {obs.notes}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      {obs.confidence !== undefined && (
                        <span className="font-mono font-bold text-emerald-700 text-[11px]">
                          {obs.confidence}%
                        </span>
                      )}
                      <span
                        className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ${
                          obs.observationType === 'CONFIRMED'
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : 'bg-rose-50 text-rose-700 border border-rose-200'
                        }`}
                      >
                        {obs.observationType}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Interactive Simulation Controls */}
              <div className="pt-2 border-t border-[#46707E]/15 flex flex-wrap items-center justify-between gap-2">
                <span className="text-[11px] font-semibold text-[#46707E]">
                  Simulate Transit Pass:
                </span>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => corroborateEvent(inspectedEvent.id, nextSimBus, true)}
                    className="px-2.5 py-1 rounded-lg text-xs font-bold bg-emerald-600 hover:bg-emerald-700 text-white transition-all shadow-sm flex items-center space-x-1"
                    title={`Simulate ${nextSimBus} passing and confirming this defect`}
                  >
                    <span>+ Corroborating Pass ({nextSimBus})</span>
                  </button>
                  <button
                    onClick={() => corroborateEvent(inspectedEvent.id, nextSimBus, false)}
                    className="px-2.5 py-1 rounded-lg text-xs font-bold bg-rose-600 hover:bg-rose-700 text-white transition-all shadow-sm flex items-center space-x-1"
                    title={`Simulate ${nextSimBus} passing without seeing defect (-12% confidence)`}
                  >
                    <span>- Non-Confirming Pass ({nextSimBus})</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Status Transition Updater */}
            <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-[#46707E]/20">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-bold text-gray-700">Update Status:</span>
                <div className="flex space-x-1.5">
                  {(['New', 'Confirmed', 'In Progress', 'Resolved'] as const).map((st) => (
                    <button
                      key={st}
                      onClick={() => updateEventStatus(inspectedEvent.id, st)}
                      className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                        inspectedEvent.status === st
                          ? 'bg-[#22393C] text-white shadow-sm'
                          : 'bg-white border border-[#46707E]/20 text-[#46707E] hover:bg-gray-100'
                      }`}
                    >
                      {st}
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={() => setSelectedEventId(null)}
                className="px-4 py-1.5 rounded-xl bg-[#46707E] text-white text-xs font-bold hover:bg-[#355560]"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

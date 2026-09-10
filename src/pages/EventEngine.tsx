import React from 'react';
import {
  Activity,
  CheckCircle,
  Clock,
  MapPin,
  Camera,
  Layers,
  ArrowRight,
  ShieldCheck,
  Send,
} from 'lucide-react';
import { useUrbanEye } from '../context/UrbanEyeContext';
import { UrbanEvent } from '../types';

export const EventEngine: React.FC = () => {
  const { generatedEngineEvent, events, setCurrentModule, navigateToEventLogWithEvent } = useUrbanEye();

  const fallbackEvt: UrbanEvent = {
    id: 'EVT-000184',
    type: 'ROAD_DAMAGE',
    category: 'Pothole',
    className: 'Pothole',
    confidence: 94,
    severity: 'HIGH',
    busId: 'BUS-01',
    timestamp: '2026-04-22 10:32:14',
    timeFormatted: '10:32 AM',
    locationName: 'MG Road, Pune',
    coordinates: [18.5204, 73.8567],
    evidenceFrame: 'evidence_EVT-000184.jpg',
    status: 'New',
    contextNote: 'Edge AI detection on front camera',
    isTransmitted: true,
  };

  // Resolve active event: prioritize any event with verified evidence
  const latestEventWithEvidence = events.find((e) => e.evidenceUrl || (e as any).evidenceDataUri);
  const evt = (generatedEngineEvent?.evidenceUrl || (generatedEngineEvent as any)?.evidenceDataUri)
    ? generatedEngineEvent
    : (latestEventWithEvidence || generatedEngineEvent || events[0] || fallbackEvt);

  const evidenceSrc = (evt as any)?.evidenceDataUri || evt?.evidenceUrl || (evt?.evidenceFrame && (evt.evidenceFrame.startsWith('/') || evt.evidenceFrame.startsWith('data:') || evt.evidenceFrame.startsWith('http')) ? evt.evidenceFrame : null);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* Top Banner / Explanation */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center space-x-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#46707E] to-[#22393C] flex items-center justify-center text-[#AFBB98] shadow-md">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-[#22393C]">Event Engine Pipeline</h2>
            <p className="text-xs text-[#46707E] font-medium mt-0.5">
              Automated Standardization: Edge AI Ingestion → Schema Normalization → Priority Dispatch
            </p>
          </div>
        </div>

        {/* Prominent Transmission Status Banner */}
        <div className="flex items-center space-x-3 bg-emerald-50 border border-emerald-300 px-4 py-2.5 rounded-xl text-emerald-900 shadow-sm">
          <CheckCircle className="w-5 h-5 text-emerald-600 shrink-0" />
          <div>
            <div className="text-xs font-extrabold uppercase tracking-wide">
              Event Transmitted to Central Platform
            </div>
            <div className="text-[10px] text-emerald-700 font-mono">
              ACK Received • MQTT Protocol v5.0 • Latency 24ms
            </div>
          </div>
        </div>
      </div>

      {/* Main Focus: Event Generated Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 7 Cols: Detailed Generated Event Card */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl space-y-5 border border-[#46707E]/25">
          <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-4">
            <div className="flex items-center space-x-3">
              <span className="w-3 h-3 rounded-full bg-rose-500 animate-ping inline-block"></span>
              <h3 className="text-lg font-extrabold text-[#22393C] tracking-wide">
                Event Generated
              </h3>
            </div>
            <span className="text-xs font-mono font-bold px-3 py-1 rounded-md bg-[#22393C] text-[#AFBB98]">
              {evt.id}
            </span>
          </div>

          {/* Evidence Frame Display */}
          <div className="relative rounded-xl overflow-hidden bg-black/90 aspect-video border border-[#46707E]/30 flex items-center justify-center group shadow-md">
            {evidenceSrc ? (
              <div className="relative w-full h-full">
                <img
                  src={evidenceSrc}
                  alt={`Evidence for ${evt.id}`}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    if ((evt as any)?.evidenceDataUri && e.currentTarget.src !== (evt as any).evidenceDataUri) {
                      e.currentTarget.src = (evt as any).evidenceDataUri;
                    } else if (evt?.evidenceUrl && e.currentTarget.src !== evt.evidenceUrl) {
                      e.currentTarget.src = evt.evidenceUrl;
                    }
                  }}
                />
                {/* AI Verified Badge */}
                <div className="absolute top-3 right-3 px-2.5 py-1 rounded-md bg-black/75 backdrop-blur-md border border-white/20 text-[10px] font-mono font-bold text-emerald-400 flex items-center space-x-1.5 shadow-md">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span>AI VERIFIED FRAME</span>
                </div>

                {/* Bottom metadata chip */}
                <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between text-[11px] font-mono text-white/90 bg-black/75 backdrop-blur-md px-3 py-1 rounded border border-white/10">
                  <span>EVIDENCE: {evt.evidenceFrame}</span>
                  <span>BUS: {evt.busId}</span>
                  <span>GPS: {evt.coordinates?.[0]?.toFixed(4) ?? '18.5204'}, {evt.coordinates?.[1]?.toFixed(4) ?? '73.8567'}</span>
                </div>
              </div>
            ) : (
              /* Visual simulated road surface with bounding box overlay fallback */
              <div className="w-full h-full bg-gradient-to-b from-[#2A3B3D] via-[#1E292B] to-[#141B1C] flex flex-col items-center justify-center p-6 relative">
                {/* Simulated asphalt texture markings */}
                <div className="absolute inset-0 opacity-20 bg-[radial-gradient(#AFBB98_1px,transparent_1px)] [background-size:16px_16px]"></div>
                
                {/* Detected Pothole Bounding Box */}
                <div className="relative border-2 border-rose-500 bg-rose-500/20 rounded p-6 text-center animate-pulse">
                  <div className="absolute -top-4 left-2 bg-rose-600 text-white text-[10px] font-mono font-bold px-2 py-0.5 rounded">
                    {evt.className} {evt.confidence}%
                  </div>
                  <div className="w-24 h-12 border border-dashed border-rose-300 rounded-full mx-auto my-1 flex items-center justify-center text-[10px] text-white/80">
                    Defect ROI
                  </div>
                </div>

                {/* Bottom metadata chip */}
                <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between text-[11px] font-mono text-white/80 bg-black/60 px-3 py-1 rounded">
                  <span>EVIDENCE: {evt.evidenceFrame}</span>
                  <span>BUS: {evt.busId}</span>
                  <span>GPS: {evt.coordinates?.[0]?.toFixed(4) ?? '18.5204'}, {evt.coordinates?.[1]?.toFixed(4) ?? '73.8567'}</span>
                </div>
              </div>
            )}
          </div>

          {/* Metadata Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 pt-2">
            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-[10px] uppercase font-bold text-gray-500">Event Type</span>
              <div className="text-sm font-mono font-bold text-[#22393C] mt-0.5">
                {evt.type}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-[10px] uppercase font-bold text-gray-500">Class</span>
              <div className="text-sm font-bold text-[#46707E] mt-0.5">
                {evt.className}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-[10px] uppercase font-bold text-gray-500">Confidence</span>
              <div className="text-sm font-mono font-extrabold text-emerald-700 mt-0.5">
                {evt.confidence}%
              </div>
            </div>

            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-[10px] uppercase font-bold text-gray-500">Severity</span>
              <div className="text-sm font-bold text-rose-700 mt-0.5">
                {evt.severity}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-[10px] uppercase font-bold text-gray-500">Bus ID</span>
              <div className="text-sm font-mono font-bold text-[#22393C] mt-0.5">
                {evt.busId}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-white/70 border border-[#46707E]/15">
              <span className="text-[10px] uppercase font-bold text-gray-500">Location</span>
              <div className="text-xs font-mono font-semibold text-[#22393C] mt-1 truncate">
                {evt.coordinates?.[0]?.toFixed(4) ?? '18.5204'}, {evt.coordinates?.[1]?.toFixed(4) ?? '73.8567'}
              </div>
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-[#22393C]/5 border border-[#46707E]/20 text-xs text-[#22393C] flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Clock className="w-4 h-4 text-[#46707E]" />
              <span>Timestamp: <strong>{evt.timestamp}</strong></span>
            </div>
            <div className="flex items-center space-x-2">
              <MapPin className="w-4 h-4 text-[#46707E]" />
              <span>{evt.locationName}</span>
            </div>
          </div>

          {/* Multi-Bus Corroboration Metadata Strip */}
          <div className="p-3 rounded-xl bg-white/80 border border-[#46707E]/20 text-xs flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center space-x-2">
              <span className="text-gray-500 font-bold uppercase text-[10px]">Multi-Bus Corroboration:</span>
              <span
                className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold ${
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
                  ? '🟢 Verified (3+ Passes)'
                  : evt.corroborationStatus === 'CORROBORATED'
                  ? '🟡 Corroborated (2 Buses)'
                  : evt.corroborationStatus === 'POTENTIAL_FALSE_POSITIVE'
                  ? '🔴 Potential False Positive'
                  : evt.corroborationStatus === 'NEEDS_VERIFICATION'
                  ? '⚠️ Needs Verification'
                  : '🔵 Single Bus Observation'}
              </span>
            </div>
            {evt.participatingBuses && evt.participatingBuses.length > 0 && (
              <div className="flex items-center space-x-1.5 text-[11px] font-mono">
                <span className="text-gray-500">Participating Fleet:</span>
                {evt.participatingBuses.map((b) => (
                  <span key={b} className="bg-[#22393C]/10 text-[#22393C] font-bold px-1.5 py-0.5 rounded border border-[#46707E]/20">
                    {b}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Action Row */}
          <div className="pt-2 flex items-center justify-between">
            <button
              onClick={() => navigateToEventLogWithEvent(evt.id)}
              className="px-5 py-2.5 rounded-xl bg-[#22393C] hover:bg-[#2C494D] text-[#CECDB9] text-xs font-bold flex items-center space-x-2 transition-all shadow-md"
            >
              <span>View in Event Log</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <span className="text-xs text-emerald-800 font-bold flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              Verified by Central Event Hub
            </span>
          </div>
        </div>

        {/* Right 5 Cols: Standardized Stream Queue */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-3">
            <h3 className="font-bold text-sm text-[#22393C]">Standardized Event Stream</h3>
            <span className="text-xs text-gray-500 font-mono">Latest Ingestions</span>
          </div>

          <div className="space-y-3">
            {events.slice(0, 5).map((e) => (
              <div
                key={e.id}
                onClick={() => navigateToEventLogWithEvent(e.id)}
                className="p-3.5 rounded-xl bg-white/75 border border-[#46707E]/15 hover:border-[#46707E]/40 hover:bg-white transition-all cursor-pointer space-y-1.5"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-[#22393C]">{e.id}</span>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                      e.severity === 'HIGH' || e.severity === 'CRITICAL'
                        ? 'bg-rose-100 text-rose-700'
                        : 'bg-emerald-100 text-emerald-700'
                    }`}
                  >
                    {e.severity}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs text-[#46707E]">
                  <span className="font-semibold">{e.className} ({e.busId})</span>
                  <span className="font-mono font-bold text-emerald-700">{e.confidence}%</span>
                </div>
                <div className="flex items-center justify-between text-[11px] text-gray-500 pt-1 border-t border-black/5">
                  <span>{e.locationName}</span>
                  <span>{e.timeFormatted}</span>
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={() => setCurrentModule('event-log')}
            className="w-full py-2 rounded-xl border border-[#46707E]/30 text-xs font-bold text-[#46707E] hover:bg-[#46707E]/10 transition-all text-center"
          >
            Open Complete Event Log →
          </button>
        </div>
      </div>
    </div>
  );
};

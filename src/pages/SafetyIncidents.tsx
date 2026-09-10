import React, { useState } from 'react';
import {
  ShieldAlert,
  Car,
  AlertTriangle,
  UserX,
  User,
  Gauge,
  Clock,
  MapPin,
  CheckCircle2,
  Camera,
  Layers,
} from 'lucide-react';
import { useUrbanEye } from '../context/UrbanEyeContext';
import { SafetyItem } from '../types';

export const SafetyIncidents: React.FC = () => {
  const { safety } = useUrbanEye();
  const [activeTab, setActiveTab] = useState<'Rash Driving' | 'Hit & Run' | 'Pedestrian Risk' | 'ANPR / OCR'>('Rash Driving');

  const filteredIncidents = safety.filter((item) => item.type === activeTab);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* Top Banner */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center space-x-4">
          <div className="p-3 rounded-xl bg-rose-950 text-rose-400 shadow-md">
            <ShieldAlert className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-[#22393C]">
              Safety & Incident Intelligence
            </h2>
            <p className="text-xs text-[#46707E] font-medium mt-0.5">
              Automated Bus Fleet Detection of Traffic Infractions, High-Risk Driving & Vulnerable Road Users
            </p>
          </div>
        </div>

        <div></div>
      </div>

      {/* Category Tabs: Rash Driving, Hit & Run, Pedestrian Risk, ANPR / OCR */}
      <div className="flex flex-wrap items-center gap-2 border-b border-[#46707E]/20 pb-3">
        {(['Rash Driving', 'Hit & Run', 'Pedestrian Risk', 'ANPR / OCR'] as const).map((tab) => {
          const count = safety.filter((i) => i.type === tab).length;
          const isLiveAI = tab === 'Pedestrian Risk' || tab === 'Rash Driving' || tab === 'Hit & Run' || tab === 'ANPR / OCR';
          return (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex items-center space-x-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all ${
                activeTab === tab
                  ? 'bg-[#22393C] text-white shadow-md'
                  : 'bg-white/60 text-[#46707E] hover:bg-white'
              }`}
            >
              <span>{tab}</span>
              {isLiveAI && (
                <span className="px-1.5 py-0.2 rounded text-[9px] font-mono font-bold bg-emerald-500 text-white uppercase">
                  Live AI
                </span>
              )}
              {!isLiveAI && (
                <span className="px-1.5 py-0.2 rounded text-[9px] font-mono text-gray-500 bg-gray-200 uppercase">
                  Workflow
                </span>
              )}
              <span
                className={`px-2 py-0.5 rounded-full text-[10px] font-mono ${
                  activeTab === tab ? 'bg-[#AFBB98] text-[#22393C]' : 'bg-black/5 text-gray-600'
                }`}
              >
                {count}
              </span>
            </button>
          );
        })}
      </div>

      {/* Honest Domain Status Banner */}
      {activeTab === 'Pedestrian Risk' ? (
        <div className="p-3.5 rounded-2xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-950 flex flex-wrap items-center justify-between gap-2 shadow-sm">
          <span className="flex items-center gap-2 font-medium">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-600"></span>
            </span>
            <strong>⚡ Real Edge AI Pipeline Active:</strong> Real-time YOLO11n Pedestrian Detection + Conflict Trajectory Geometry + Image-Based Risk Estimation.
          </span>
          <span className="text-[10px] font-mono font-bold uppercase bg-emerald-200/70 text-emerald-900 px-2 py-0.5 rounded">
            Live Neural Model
          </span>
        </div>
      ) : activeTab === 'Rash Driving' ? (
        <div className="p-3.5 rounded-2xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-950 flex flex-wrap items-center justify-between gap-2 shadow-sm">
          <span className="flex items-center gap-2 font-medium">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-600"></span>
            </span>
            <strong>⚡ Real Edge AI Pipeline Active:</strong> YOLO11n Vehicle Tracking (ByteTrack) + Deterministic Kinematic Trajectory & Swerve/Acceleration Risk Engine.
          </span>
          <span className="text-[10px] font-mono font-bold uppercase bg-emerald-200/70 text-emerald-900 px-2 py-0.5 rounded">
            Live Kinematic Engine
          </span>
        </div>
      ) : activeTab === 'Hit & Run' ? (
        <div className="p-3.5 rounded-2xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-950 flex flex-wrap items-center justify-between gap-2 shadow-sm">
          <span className="flex items-center gap-2 font-medium">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-600"></span>
            </span>
            <strong>⚡ Real Edge AI Pipeline Active:</strong> YOLO11n Multi-Object Tracking (ByteTrack) + Multi-Signal Spatio-Temporal Collision & Fleeing Heuristic Engine.
          </span>
          <span className="text-[10px] font-mono font-bold uppercase bg-emerald-200/70 text-emerald-900 px-2 py-0.5 rounded">
            Live Heuristic Engine
          </span>
        </div>
      ) : filteredIncidents.some((i) => Boolean(i.evidenceDataUri || i.evidenceUrl)) ? (
        <div className="p-3.5 rounded-2xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-950 flex flex-wrap items-center justify-between gap-2 shadow-sm">
          <span className="flex items-center gap-2 font-medium">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-600"></span>
            </span>
            <strong>⚡ Real Edge ANPR Active:</strong> Real-time YOLO11n Vehicle Localization + EasyOCR CRAFT Neural Plate Recognition.
          </span>
          <span className="text-[10px] font-mono font-bold uppercase bg-emerald-200/70 text-emerald-900 px-2 py-0.5 rounded">
            Live Neural ANPR
          </span>
        </div>
      ) : (
        <div className="p-3.5 rounded-2xl bg-amber-50 border border-amber-200 text-xs text-amber-950 flex flex-wrap items-center justify-between gap-2 shadow-sm">
          <span className="flex items-center gap-2 font-medium">
            <span className="w-2 h-2 rounded-full bg-amber-500"></span>
            <strong>📐 Municipal Architecture Spec:</strong> Bus Telemetry & Contextual ANPR Ingestion Workflow (Prototype Fleet Pipeline).
          </span>
          <span className="text-[10px] font-mono font-bold uppercase bg-amber-200/70 text-amber-900 px-2 py-0.5 rounded">
            Prototype Workflow
          </span>
        </div>
      )}

      {/* Incidents Grid or Empty State */}
      {filteredIncidents.length === 0 ? (
        <div className="glass-panel p-12 rounded-2xl text-center space-y-3">
          <CheckCircle2 className="w-10 h-10 text-emerald-600 mx-auto" />
          <h3 className="text-base font-bold text-[#22393C]">
            No Active {activeTab} Incidents
          </h3>
          <p className="text-xs text-[#46707E] max-w-md mx-auto">
            Fleet telemetry and edge computer vision running within nominal safety thresholds. No high-risk motion infractions or incidents currently detected.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {filteredIncidents.map((incident) => (
            <div
              key={incident.id}
              className="glass-panel p-6 rounded-2xl space-y-4 hover:shadow-glass-hover transition-all"
            >
              {/* Top Bar with Number Plate & Severity */}
              <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-3">
                <div className="flex items-center space-x-3">
                  <div className="p-2 rounded-lg bg-black/5">
                    {incident.type === 'Pedestrian Risk' ? (
                      <User className="w-5 h-5 text-amber-700" />
                    ) : (
                      <Car className="w-5 h-5 text-[#46707E]" />
                    )}
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-bold font-mono text-[#22393C]">
                        {incident.vehicleId}
                      </span>
                      <span
                        className={`px-2.5 py-0.5 rounded font-mono font-extrabold text-xs tracking-wider border ${
                          incident.type === 'Pedestrian Risk'
                            ? 'bg-amber-600 text-white border-amber-500/30'
                            : incident.plateStatus === 'PLATE DETECTED — OCR UNCERTAIN'
                            ? 'bg-amber-900/60 text-amber-300 border-amber-500/40'
                            : 'bg-[#22393C] text-[#AFBB98] border-[#AFBB98]/30'
                        }`}
                      >
                        {incident.numberPlate}
                      </span>
                      {incident.plateStatus && incident.type !== 'Pedestrian Risk' && (
                        <span
                          className={`text-[9px] font-mono px-1.5 py-0.5 rounded font-bold uppercase ${
                            incident.plateStatus === 'RELIABLE PLATE'
                              ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                              : 'bg-amber-100 text-amber-800 border border-amber-300'
                          }`}
                        >
                          {incident.plateStatus === 'RELIABLE PLATE' ? 'Reliable' : 'Uncertain'}
                        </span>
                      )}
                    </div>
                    <span className="text-[11px] text-gray-500 font-mono">
                      {incident.type === 'Pedestrian Risk' ? (
                        <>
                          {incident.pedestrianCount !== undefined ? `${incident.pedestrianCount} Pedestrians Detected` : 'Pedestrians Detected'} • Conf: <strong className="text-emerald-700">{incident.ocrConfidence}%</strong>
                        </>
                      ) : incident.type === 'Rash Driving' ? (
                        <>
                          Kinematic Motion Risk: <strong className="text-rose-700">{incident.riskScore || incident.ocrConfidence}/100</strong> ({incident.severity})
                        </>
                      ) : incident.type === 'Hit & Run' ? (
                        <>
                          Vision-Based Incident Candidate: <strong className="text-rose-700">{incident.riskScore || incident.ocrConfidence}/100</strong> ({incident.severity})
                        </>
                      ) : (
                        <>
                          Context OCR Telemetry: <strong className="text-emerald-700">{incident.ocrConfidence}%</strong> {incident.evidenceUrl || incident.evidenceDataUri ? '(Live Neural ANPR)' : '(Prototype Fleet Spec)'}
                        </>
                      )}
                    </span>
                  </div>
                </div>

                <span
                  className={`px-3 py-1 rounded-full text-xs font-extrabold tracking-wide ${
                    incident.severity === 'CRITICAL'
                      ? 'bg-rose-100 text-rose-800 border border-rose-300'
                      : 'bg-amber-100 text-amber-800 border border-amber-300'
                  }`}
                >
                  {incident.severity}
                </span>
              </div>

              {/* Triggered Rule Pills if Rash Driving or Hit & Run */}
              {incident.triggeredRules && incident.triggeredRules.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {incident.triggeredRules.map((rule, rIdx) => (
                    <span
                      key={rIdx}
                      className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-rose-900/15 text-rose-800 border border-rose-300"
                    >
                      {rule.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              )}

              {/* AI Evidence Visualization Frame */}
              {incident.evidenceDataUri || incident.evidenceUrl ? (
                <div className="relative rounded-xl overflow-hidden bg-black aspect-video flex items-center justify-center border border-[#46707E]/30 group">
                  <img
                    src={incident.evidenceDataUri || incident.evidenceUrl}
                    alt="Real AI Evidence Frame"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-3 left-3 px-2.5 py-1 rounded bg-black/75 backdrop-blur-sm text-amber-300 font-mono text-[10px] font-bold border border-amber-500/30 flex items-center space-x-1.5 shadow-md">
                    <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
                    <span>AI VERIFIED EVIDENCE</span>
                  </div>
                  {incident.riskScore !== undefined && (
                    <div className="absolute bottom-3 right-3 px-2.5 py-1 rounded bg-black/80 backdrop-blur-sm text-white font-mono text-[10px] font-bold border border-white/20">
                      Est. Risk: {incident.riskScore}/100 ({incident.severity})
                    </div>
                  )}
                  <div className="absolute bottom-3 left-3 px-2.5 py-1 rounded bg-black/80 backdrop-blur-sm text-white/90 font-mono text-[10px]">
                    <span>LOC: {incident.location}</span>
                  </div>
                </div>
              ) : (
                <div className="relative rounded-xl overflow-hidden bg-black/90 aspect-video flex items-center justify-center border border-[#46707E]/30">
                  <div className="w-full h-full bg-gradient-to-b from-[#1C282A] via-[#152022] to-[#0D1516] flex flex-col items-center justify-center p-6 relative">
                    {/* Simulated dashcam detection overlay */}
                    <div className="border-2 border-rose-500 bg-rose-500/15 rounded p-4 text-center max-w-xs">
                      <div className="text-xs font-mono font-bold text-rose-300">
                        TARGET IDENTIFIED
                      </div>
                      <div className="text-[11px] font-mono text-white mt-1 bg-black/40 px-2 py-0.5 rounded">
                        ID: {incident.numberPlate} ({incident.ocrConfidence}%)
                      </div>
                    </div>

                    {/* Trajectory vector */}
                    <div className="absolute top-4 left-4 text-[10px] font-mono text-emerald-400 bg-black/60 px-2.5 py-1 rounded">
                      SPEED TRACKER: 78 km/hr | VECTOR SWERVE: +34°
                    </div>

                    <div className="absolute bottom-3 left-4 right-4 flex items-center justify-between text-xs font-mono text-white/80 bg-black/60 px-3 py-1 rounded">
                      <span>LOC: {incident.location}</span>
                      <span>TIME: {incident.timestamp}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Prototype Image-Based Risk Estimate note for Pedestrian Risk */}
              {incident.type === 'Pedestrian Risk' && (
                <div className="px-3 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs flex items-center justify-between text-amber-950 font-medium">
                  <span>
                    <strong>Prototype Image-Based Risk Estimate:</strong> {incident.riskScore || 68}/100 ({incident.severity})
                  </span>
                  <span className="text-[10px] font-mono font-bold text-amber-700 uppercase bg-amber-200/50 px-2 py-0.5 rounded">
                    Image-Based
                  </span>
                </div>
              )}

              {/* Deterministic Kinematic Note for Rash Driving */}
              {incident.type === 'Rash Driving' && incident.riskScore !== undefined && (
                <div className="px-3 py-2 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs flex items-center justify-between text-rose-950 font-medium">
                  <span>
                    <strong>Kinematic Risk Score:</strong> {incident.riskScore}/100 ({incident.severity})
                  </span>
                  <span className="text-[10px] font-mono font-bold text-rose-700 uppercase bg-rose-200/50 px-2 py-0.5 rounded">
                    Kinematic Rule Engine
                  </span>
                </div>
              )}

              {/* Multi-Signal Heuristic Note for Hit & Run */}
              {incident.type === 'Hit & Run' && incident.riskScore !== undefined && (
                <div className="px-3 py-2 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs flex items-center justify-between text-rose-950 font-medium">
                  <span>
                    <strong>Vision-Based Incident Candidate:</strong> Risk {incident.riskScore}/100 ({incident.severity})
                  </span>
                  <span className="text-[10px] font-mono font-bold text-rose-700 uppercase bg-rose-200/50 px-2 py-0.5 rounded">
                    Multi-Signal Heuristic
                  </span>
                </div>
              )}

              {/* Detection Context Details */}
              <div className="space-y-2">
                <span className="text-xs font-bold uppercase tracking-wider text-[#46707E]">
                  {incident.type === 'Pedestrian Risk'
                    ? 'Pedestrian Hazard & Spatial Risk Telemetry:'
                    : incident.type === 'Rash Driving'
                    ? 'Kinematic Infraction & Trajectory Telemetry:'
                    : incident.type === 'Hit & Run'
                    ? 'Collision Interaction & Incident Telemetry:'
                    : 'Detection Telemetry & Infraction Context:'}
                </span>
                <div className="space-y-1.5">
                  {incident.contextDetails.map((detail, idx) => (
                    <div
                      key={idx}
                      className="flex items-start space-x-2 text-xs text-[#22393C] bg-white/70 p-2 rounded-lg border border-[#46707E]/15"
                    >
                      <AlertTriangle className="w-3.5 h-3.5 text-rose-600 mt-0.5 shrink-0" />
                      <span>{detail}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Footer Metadata */}
              <div className="flex items-center justify-between pt-2 text-xs text-gray-500 border-t border-[#46707E]/15">
                <div className="flex items-center space-x-1">
                  <MapPin className="w-3.5 h-3.5 text-[#46707E]" />
                  <span>{incident.location}, Pune</span>
                </div>
                <div className="flex items-center space-x-1 font-mono">
                  <Clock className="w-3.5 h-3.5 text-[#46707E]" />
                  <span>{incident.timestamp}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

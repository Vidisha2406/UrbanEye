import React, { useMemo } from 'react';
import {
  Gauge,
  AlertTriangle,
  Wrench,
  TrendingUp,
  Bus,
  Layers,
  Sparkles,
  ClipboardList,
  Droplets,
  Footprints,
  Split,
  TrafficCone,
  Radio,
} from 'lucide-react';
import { useUrbanEye } from '../context/UrbanEyeContext';

export const Infrastructure: React.FC = () => {
  const { events } = useUrbanEye();

  // Dynamic live event counts by infrastructure category
  const liveRoadSurface = useMemo(
    () => events.filter((e) => e.type === 'ROAD_DAMAGE' || e.category === 'Road Damage' || e.category === 'Pothole').length,
    [events]
  );
  const liveDrainage = useMemo(
    () => events.filter((e) => e.type === 'WATERLOGGING' || e.category === 'Waterlogging').length,
    [events]
  );
  const liveCrossings = useMemo(
    () => events.filter((e) => e.type === 'ZEBRA_CROSSING' || (e.className && e.className.toLowerCase().includes('zebra'))).length,
    [events]
  );
  const liveDividers = useMemo(
    () => events.filter((e) => (e.type === 'DIVIDER' || (e.className && e.className.toLowerCase().includes('divider')))).length,
    [events]
  );
  const liveSigns = useMemo(
    () => events.filter((e) => (
      e.type === 'SIGNBOARD' ||
      e.type === 'INFRASTRUCTURE_DEFICIENCY' ||
      (e.className && e.className.toLowerCase().includes('sign')) ||
      (e.className && e.className.toLowerCase().includes('damaged'))
    )).length,
    [events]
  );

  // Dynamic live priority triage
  const liveCritical = useMemo(() => events.filter((e) => e.severity === 'CRITICAL').length, [events]);
  const liveHigh = useMemo(() => events.filter((e) => e.severity === 'HIGH').length, [events]);
  const liveMedium = useMemo(() => events.filter((e) => e.severity === 'MEDIUM').length, [events]);
  const liveLow = useMemo(() => events.filter((e) => e.severity === 'LOW').length, [events]);

  return (
    <div className="p-6 lg:p-8 max-w-6xl mx-auto space-y-8">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-[#46707E]/20 pb-5">
        <div>
          <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-[#46707E] mb-1">
            <Layers className="w-4 h-4 text-[#46707E]" />
            <span>Mobile Urban Sensing Telemetry</span>
          </div>
          <h1 className="text-2xl lg:text-3xl font-black text-[#22393C] tracking-tight">
            Road & Infrastructure Intelligence
          </h1>
          <p className="text-xs text-gray-600 mt-1 max-w-2xl">
            Autonomous condition assessment and maintenance prioritization powered by computer vision across public bus routes.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 self-start sm:self-auto">
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold bg-emerald-50 text-emerald-800 border border-emerald-300 shadow-sm">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            {events.length} Live Edge Detections
          </span>
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold bg-[#AFBB98]/25 text-[#22393C] border border-[#AFBB98]/40">
            Municipal Baseline Survey
          </span>
        </div>
      </div>

      {/* =========================================================
          1. 🛣️ ROAD SURFACE QUALITY — HERO SECTION
      ========================================================= */}
      <section className="glass-panel p-6 lg:p-8 rounded-3xl space-y-6 shadow-glass relative overflow-hidden border border-[#46707E]/25">
        {/* Subtle decorative glow */}
        <div className="absolute top-0 right-0 w-80 h-80 bg-[#AFBB98]/15 rounded-full blur-3xl pointer-events-none -mr-16 -mt-16"></div>

        <div className="flex items-center justify-between border-b border-[#46707E]/15 pb-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-2xl bg-[#46707E]/15 flex items-center justify-center text-[#22393C]">
              <Gauge className="w-5 h-5 text-[#46707E]" />
            </div>
            <div>
              <span className="text-[10px] font-extrabold uppercase tracking-widest text-[#46707E]">
                Hero Assessment
              </span>
              <h2 className="text-lg lg:text-xl font-black text-[#22393C]">
                🛣️ Road Surface Quality
              </h2>
            </div>
          </div>

          <span className="text-xs font-semibold text-gray-500 hidden sm:inline-block">
            City Pavement Index
          </span>
        </div>

        {/* Hero Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          {/* Main Overall Quality Card */}
          <div className="lg:col-span-5 bg-white/80 backdrop-blur-md rounded-2xl p-6 border border-[#46707E]/20 shadow-sm flex flex-col justify-between space-y-5">
            <div>
              <span className="text-xs font-extrabold uppercase tracking-wider text-[#46707E]">
                Overall Road Surface Quality
              </span>
              <div className="mt-3 flex items-baseline gap-3">
                <div className="text-6xl font-black font-mono text-[#22393C] tracking-tight">
                  74%
                </div>
                <div>
                  <span className="inline-block px-2.5 py-0.5 rounded-full text-xs font-extrabold bg-[#AFBB98]/30 text-[#22393C]">
                    Moderate Quality
                  </span>
                  <p className="text-[11px] text-gray-500 mt-1">Overall road-surface quality score</p>
                </div>
              </div>
            </div>

            {/* Condition Distribution Multi-segment Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-[11px] font-bold text-gray-600">
                <span>Pavement Condition Breakdown</span>
                <span className="font-mono text-[#22393C]">100%</span>
              </div>
              <div className="h-3 w-full rounded-full overflow-hidden flex bg-gray-200 border border-black/5">
                <div style={{ width: '42%' }} className="bg-[#6B8B81]" title="Good: 42%"></div>
                <div style={{ width: '35%' }} className="bg-[#46707E]" title="Moderate: 35%"></div>
                <div style={{ width: '18%' }} className="bg-amber-600" title="Poor: 18%"></div>
                <div style={{ width: '5%' }} className="bg-rose-600" title="Critical: 5%"></div>
              </div>
            </div>

            {/* Four Condition Indicators */}
            <div className="space-y-2 pt-2 border-t border-gray-100">
              <div className="grid grid-cols-2 gap-2.5 text-xs">
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#6B8B81]"></span>
                  <span className="text-gray-600 font-medium">Good:</span>
                  <span className="font-mono font-bold text-[#22393C]">42%</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#46707E]"></span>
                  <span className="text-gray-600 font-medium">Moderate:</span>
                  <span className="font-mono font-bold text-[#22393C]">35%</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-600"></span>
                  <span className="text-gray-600 font-medium">Poor:</span>
                  <span className="font-mono font-bold text-amber-700">18%</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-600"></span>
                  <span className="text-gray-600 font-medium">Critical:</span>
                  <span className="font-mono font-bold text-rose-700">5%</span>
                </div>
              </div>
              <p className="text-[10px] text-gray-500 leading-tight pt-1">
                Percentages represent condition distribution of observed road segments, while 74% is the overall road-surface quality score.
              </p>
            </div>
          </div>

          {/* 3 Compact Assessment Metrics */}
          <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Metric 1: Pothole Density */}
            <div className="bg-white/70 backdrop-blur-sm p-5 rounded-2xl border border-[#46707E]/15 hover:border-[#46707E]/35 transition-all flex flex-col justify-between space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-[#46707E]">
                  Pothole Density
                </span>
                <AlertTriangle className="w-4 h-4 text-amber-600" />
              </div>
              <div>
                <div className="text-3xl font-mono font-extrabold text-[#22393C]">
                  2.3
                </div>
                <p className="text-[11px] text-gray-500 mt-1">potholes / km</p>
              </div>
              <div className="space-y-1">
                <div className="w-full bg-gray-200 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-amber-600 h-full rounded-full" style={{ width: '46%' }}></div>
                </div>
                <div className="text-[10px] text-gray-400">Arterial transit routes</div>
              </div>
            </div>

            {/* Metric 2: Surface Damage */}
            <div className="bg-white/70 backdrop-blur-sm p-5 rounded-2xl border border-[#46707E]/15 hover:border-[#46707E]/35 transition-all flex flex-col justify-between space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-[#46707E]">
                  Surface Damage
                </span>
                <Layers className="w-4 h-4 text-[#46707E]" />
              </div>
              <div>
                <div className="text-3xl font-mono font-extrabold text-[#22393C]">
                  18.4%
                </div>
                <p className="text-[11px] text-gray-500 mt-1">area affected</p>
              </div>
              <div className="space-y-1">
                <div className="w-full bg-gray-200 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-[#46707E] h-full rounded-full" style={{ width: '37%' }}></div>
                </div>
                <div className="text-[10px] text-gray-400">Ravelling & rutting</div>
              </div>
            </div>

            {/* Metric 3: Deterioration Trend */}
            <div className="bg-white/70 backdrop-blur-sm p-5 rounded-2xl border border-[#46707E]/15 hover:border-[#46707E]/35 transition-all flex flex-col justify-between space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-[#46707E]">
                  Deterioration Trend
                </span>
                <TrendingUp className="w-4 h-4 text-rose-600" />
              </div>
              <div>
                <div className="text-3xl font-mono font-extrabold text-rose-700">
                  +8.2%
                </div>
                <p className="text-[11px] text-gray-500 mt-1">quarterly rate</p>
              </div>
              <div className="space-y-1">
                <div className="w-full bg-gray-200 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-rose-600 h-full rounded-full" style={{ width: '65%' }}></div>
                </div>
                <div className="text-[10px] text-rose-600 font-semibold">Seasonal wear</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================
          2. 📋 INFRASTRUCTURE CONDITION SNAPSHOT
      ========================================================= */}
      <section className="glass-panel p-6 lg:p-8 rounded-3xl space-y-5 border border-[#46707E]/20 shadow-glass">
        <div className="flex items-center justify-between border-b border-[#46707E]/15 pb-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-2xl bg-[#46707E]/15 flex items-center justify-center text-[#22393C]">
              <ClipboardList className="w-5 h-5 text-[#46707E]" />
            </div>
            <div>
              <span className="text-[10px] font-extrabold uppercase tracking-widest text-[#46707E]">
                Asset Overview
              </span>
              <h2 className="text-lg lg:text-xl font-black text-[#22393C]">
                📋 Infrastructure Condition Snapshot
              </h2>
            </div>
          </div>

          <span className="text-xs font-semibold text-gray-500">
            5 Monitored Categories
          </span>
        </div>

        {/* Clean Snapshot Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-gray-200/80 text-gray-500 font-extrabold uppercase tracking-wider text-[10px]">
                <th className="py-3 px-4">Category</th>
                <th className="py-3 px-4">Live Edge AI</th>
                <th className="py-3 px-4">Baseline Survey</th>
                <th className="py-3 px-4">Condition</th>
                <th className="py-3 px-4 text-right">Priority</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100/90 font-medium">
              {[
                {
                  category: 'Road Surface',
                  icon: Layers,
                  liveCount: liveRoadSurface,
                  baselineIssues: 182,
                  condition: 'Moderate',
                  conditionStyle: 'bg-blue-50 text-blue-800 border-blue-200',
                  priority: 'High',
                  priorityStyle: 'bg-amber-100 text-amber-800 border-amber-300',
                  hasEdgeModel: true,
                },
                {
                  category: 'Drainage',
                  icon: Droplets,
                  liveCount: liveDrainage,
                  baselineIssues: 51,
                  condition: 'Needs Attention',
                  conditionStyle: 'bg-amber-50 text-amber-800 border-amber-200',
                  priority: 'High',
                  priorityStyle: 'bg-amber-100 text-amber-800 border-amber-300',
                  hasEdgeModel: false,
                },
                {
                  category: 'Zebra Crossings',
                  icon: Footprints,
                  liveCount: liveCrossings,
                  baselineIssues: 24,
                  condition: 'Fair',
                  conditionStyle: 'bg-[#6B8B81]/15 text-[#22393C] border-[#6B8B81]/30',
                  priority: 'Medium',
                  priorityStyle: 'bg-blue-50 text-blue-800 border-blue-200',
                  hasEdgeModel: false,
                },
                {
                  category: 'Road Dividers',
                  icon: Split,
                  liveCount: liveDividers,
                  baselineIssues: 29,
                  condition: 'Moderate',
                  conditionStyle: 'bg-blue-50 text-blue-800 border-blue-200',
                  priority: 'Medium',
                  priorityStyle: 'bg-blue-50 text-blue-800 border-blue-200',
                  hasEdgeModel: false,
                },
                {
                  category: 'Traffic Signboards',
                  icon: TrafficCone,
                  liveCount: liveSigns,
                  baselineIssues: 18,
                  condition: 'Poor',
                  conditionStyle: 'bg-rose-50 text-rose-800 border-rose-200',
                  priority: 'Critical',
                  priorityStyle: 'bg-rose-100 text-rose-800 border-rose-300',
                  hasEdgeModel: true,
                },
              ].map((item, idx) => {
                const IconComponent = item.icon;
                return (
                  <tr key={idx} className="hover:bg-white/70 transition-colors">
                    <td className="py-3.5 px-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2.5">
                        <div className="w-7 h-7 rounded-lg bg-[#46707E]/10 flex items-center justify-center text-[#46707E]">
                          <IconComponent className="w-3.5 h-3.5" />
                        </div>
                        <span className="font-extrabold text-[#22393C] text-sm">
                          {item.category}
                        </span>
                      </div>
                    </td>

                    <td className="py-3.5 px-4 whitespace-nowrap">
                      {item.hasEdgeModel ? (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded font-mono font-bold text-xs bg-emerald-50 text-emerald-800 border border-emerald-200">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                          {item.liveCount} live
                        </span>
                      ) : (
                        <span
                          className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-mono font-medium text-[11px] bg-gray-50 text-gray-500 border border-gray-200/80"
                          title="Prototype Baseline Context (No Edge AI Model)"
                        >
                          {item.liveCount > 0 ? `${item.liveCount} live` : 'Baseline Only'}
                        </span>
                      )}
                    </td>

                    <td className="py-3.5 px-4 whitespace-nowrap">
                      <span className="font-mono font-bold text-sm text-[#22393C]">
                        {item.baselineIssues}
                      </span>
                      <span className="text-[10px] text-gray-400 ml-1">survey defects</span>
                    </td>

                    <td className="py-3.5 px-4 whitespace-nowrap">
                      <span
                        className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-bold border ${item.conditionStyle}`}
                      >
                        {item.condition}
                      </span>
                    </td>

                    <td className="py-3.5 px-4 text-right whitespace-nowrap">
                      <span
                        className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-extrabold border ${item.priorityStyle}`}
                      >
                        {item.priority}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* =========================================================
          3. 🔧 MAINTENANCE PRIORITY
      ========================================================= */}
      <section className="glass-panel p-6 lg:p-8 rounded-3xl space-y-6 border border-[#46707E]/20 shadow-glass">
        <div className="flex items-center justify-between border-b border-[#46707E]/15 pb-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-2xl bg-amber-500/10 flex items-center justify-center text-amber-700">
              <Wrench className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] font-extrabold uppercase tracking-widest text-[#46707E]">
                Triage & Queue
              </span>
              <h2 className="text-lg lg:text-xl font-black text-[#22393C]">
                🔧 Maintenance Priority
              </h2>
            </div>
          </div>

          <span className="text-xs font-semibold text-gray-500">
            Action Triage
          </span>
        </div>

        {/* Priority Counts */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="p-4 rounded-2xl bg-rose-50/90 border border-rose-200 text-center">
            <div className="text-[11px] font-extrabold uppercase tracking-wider text-rose-700">
              Critical
            </div>
            <div className="text-3xl font-black font-mono text-rose-700 mt-1">
              {liveCritical} <span className="text-xs font-sans font-medium text-rose-500">Live</span> / 12 <span className="text-[10px] font-sans text-gray-400">Survey</span>
            </div>
            <div className="text-[10px] text-rose-600 font-semibold mt-0.5">Immediate Triage</div>
          </div>

          <div className="p-4 rounded-2xl bg-amber-50/90 border border-amber-200 text-center">
            <div className="text-[11px] font-extrabold uppercase tracking-wider text-amber-700">
              High
            </div>
            <div className="text-3xl font-black font-mono text-amber-700 mt-1">
              {liveHigh} <span className="text-xs font-sans font-medium text-amber-600">Live</span> / 29 <span className="text-[10px] font-sans text-gray-400">Survey</span>
            </div>
            <div className="text-[10px] text-amber-600 font-semibold mt-0.5">Priority Action</div>
          </div>

          <div className="p-4 rounded-2xl bg-[#46707E]/10 border border-[#46707E]/25 text-center">
            <div className="text-[11px] font-extrabold uppercase tracking-wider text-[#46707E]">
              Medium
            </div>
            <div className="text-3xl font-black font-mono text-[#46707E] mt-1">
              {liveMedium} <span className="text-xs font-sans font-medium text-[#46707E]">Live</span> / 47 <span className="text-[10px] font-sans text-gray-400">Survey</span>
            </div>
            <div className="text-[10px] text-gray-600 font-semibold mt-0.5">Scheduled Queue</div>
          </div>

          <div className="p-4 rounded-2xl bg-[#6B8B81]/10 border border-[#6B8B81]/25 text-center">
            <div className="text-[11px] font-extrabold uppercase tracking-wider text-[#6B8B81]">
              Low
            </div>
            <div className="text-3xl font-black font-mono text-[#6B8B81] mt-1">
              {liveLow} <span className="text-xs font-sans font-medium text-[#6B8B81]">Live</span> / 64 <span className="text-[10px] font-sans text-gray-400">Survey</span>
            </div>
            <div className="text-[10px] text-gray-600 font-semibold mt-0.5">Routine Monitoring</div>
          </div>
        </div>

        {/* Maintenance Categories */}
        <div className="space-y-3 pt-2">
          <div className="text-xs font-extrabold uppercase tracking-wider text-gray-500">
            Maintenance Categories
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {[
              {
                name: 'Road Surface Repair',
                tag: 'Urgent',
                tagColor: 'bg-rose-100 text-rose-800 border-rose-200',
              },
              {
                name: 'Drainage Maintenance',
                tag: 'High',
                tagColor: 'bg-amber-100 text-amber-800 border-amber-200',
              },
              {
                name: 'Divider Repair',
                tag: 'Moderate',
                tagColor: 'bg-blue-100 text-blue-800 border-blue-200',
              },
              {
                name: 'Zebra Crossing Repaint',
                tag: 'Scheduled',
                tagColor: 'bg-gray-100 text-gray-700 border-gray-200',
              },
              {
                name: 'Signal Infrastructure',
                tag: 'High',
                tagColor: 'bg-amber-100 text-amber-800 border-amber-200',
              },
            ].map((cat, idx) => (
              <div
                key={idx}
                className="bg-white/75 p-4 rounded-2xl border border-gray-200/80 hover:bg-white hover:border-[#46707E]/30 transition-all flex flex-col justify-between space-y-2"
              >
                <span className="font-bold text-xs text-[#22393C] leading-snug">
                  {cat.name}
                </span>
                <div>
                  <span
                    className={`inline-block text-[10px] font-extrabold uppercase px-2 py-0.5 rounded-full border ${cat.tagColor}`}
                  >
                    {cat.tag}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* =========================================================
          4. 🚌 BUS ROUTE INSPECTION COVERAGE
      ========================================================= */}
      <section className="glass-panel p-6 lg:p-8 rounded-3xl space-y-6 border border-[#46707E]/20 shadow-glass">
        <div className="flex items-center justify-between border-b border-[#46707E]/15 pb-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-2xl bg-[#22393C]/10 flex items-center justify-center text-[#22393C]">
              <Bus className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] font-extrabold uppercase tracking-widest text-[#46707E]">
                Mobile Urban Sensing
              </span>
              <h2 className="text-lg lg:text-xl font-black text-[#22393C]">
                🚌 Bus Route Inspection Coverage
              </h2>
            </div>
          </div>

          <span className="text-xs font-mono font-bold text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-200">
            Active Mobile Fleet Sensing
          </span>
        </div>

        {/* 3 Summary Statistics Counters */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="p-4 bg-white/75 rounded-2xl border border-gray-200/80 text-center">
            <div className="text-[11px] font-extrabold uppercase text-gray-500">
              Routes Inspected
            </div>
            <div className="text-3xl font-mono font-black text-[#22393C] mt-1">
              18 / 25
            </div>
            <div className="text-[10px] text-gray-400 mt-1">Fleet Coverage</div>
          </div>

          <div className="p-4 bg-white/75 rounded-2xl border border-gray-200/80 text-center">
            <div className="text-[11px] font-extrabold uppercase text-gray-500">
              Road KM Observed
            </div>
            <div className="text-3xl font-mono font-black text-[#46707E] mt-1">
              142 km
            </div>
            <div className="text-[10px] text-gray-400 mt-1">Daily Transit Sweep</div>
          </div>

          <div className="p-4 bg-white/75 rounded-2xl border border-amber-200 text-center">
            <div className="text-[11px] font-extrabold uppercase text-amber-700">
              Total AI Detections
            </div>
            <div className="text-3xl font-mono font-black text-amber-700 mt-1">
              {events.length} <span className="text-sm font-sans font-bold text-[#22393C]">Live</span> / 326 <span className="text-xs font-sans font-normal text-gray-500">Archive</span>
            </div>
            <div className="text-[10px] text-gray-500 mt-1">Active Edge Ingestion + Baseline Municipal Archive</div>
          </div>
        </div>

        {/* 4 Representative Route Coverage Bars */}
        <div className="space-y-3 pt-2">
          <div className="text-xs font-extrabold uppercase tracking-wider text-gray-500">
            Representative Route Coverage
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[
              {
                route: 'Route 17',
                coverage: 94,
                color: '#6B8B81',
              },
              {
                route: 'Route 11',
                coverage: 89,
                color: '#46707E',
              },
              {
                route: 'Route 22',
                coverage: 81,
                color: '#AFBB98',
              },
              {
                route: 'Route 9',
                coverage: 63,
                color: '#22393C',
              },
            ].map((item, idx) => (
              <div
                key={idx}
                className="bg-white/75 p-4 rounded-2xl border border-gray-200/80 space-y-2 hover:bg-white transition-all"
              >
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-2">
                    <span className="font-extrabold text-[#22393C] text-sm">
                      {item.route}
                    </span>
                    <span className="text-[10px] text-gray-400 uppercase font-semibold">
                      Transit Corridor
                    </span>
                  </div>
                  <span className="font-mono font-black text-base text-[#22393C]">
                    {item.coverage}%
                  </span>
                </div>

                <div className="w-full bg-gray-200 h-2.5 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{ width: `${item.coverage}%`, backgroundColor: item.color }}
                  ></div>
                </div>

                <div className="flex items-center justify-between text-[10px] text-gray-500 pt-0.5">
                  <span>Inspection Sweep Complete</span>
                  <span className="font-medium text-[#46707E]">AI Active</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom Concept Tagline */}
        <div className="p-3.5 rounded-2xl bg-[#46707E]/10 border border-[#46707E]/20 text-xs text-[#22393C] flex items-center justify-between">
          <span className="flex items-center gap-2 font-medium">
            <Sparkles className="w-4 h-4 text-[#46707E]" />
            Public transport buses act as mobile urban sensing units across municipal road networks.
          </span>
          <span className="font-bold text-[11px] text-[#46707E] hidden md:inline">
            PS 26124
          </span>
        </div>
      </section>
    </div>
  );
};

import React, { useState, useMemo } from 'react';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Flame,
  CheckSquare,
  Square,
  BarChart3,
  MapPin,
  Lightbulb,
  Compass,
  ArrowRight,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { LightCityMap } from '../components/maps/LightCityMap';
import { useUrbanEye } from '../context/UrbanEyeContext';
import { MapLayerFilterState } from '../types';
import { analyticsData } from '../data/mockData';

export const Dashboard: React.FC = () => {
  const { buses, events, categoryStats, navigateToEventLogWithEvent, setCurrentModule } = useUrbanEye();

  // Dynamically compute Events by Type from real events
  const dynamicEventsByType = useMemo(() => {
    const categoryColors: Record<string, string> = {
      'Road Damage': '#46707E',
      'Pothole': '#46707E',
      'Traffic': '#546F67',
      'Vehicle': '#22393C',
      'Safety / Pedestrian Risk': '#AFBB98',
      'Accidents / Incidents': '#B91C1C',
      'Waterlogging': '#6B8B81',
      'Infrastructure Deficiencies': '#335055',
      'ANPR / OCR Events': '#D97706',
    };
    const counts: Record<string, number> = {};
    events.forEach((e) => {
      const cat = e.category || 'Other';
      counts[cat] = (counts[cat] || 0) + 1;
    });
    return Object.entries(counts).map(([name, value]) => ({
      name,
      value,
      color: categoryColors[name] || '#6B8B81',
    }));
  }, [events]);

  // Dynamically compute category stats from real events overlayed on base stats
  const dynamicCategoryStats = useMemo(() => {
    const categoryCounts: Record<string, number> = {};
    events.forEach((e) => {
      const cat = e.category || 'Other';
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
    });

    return categoryStats.map((stat) => {
      const eventCount = categoryCounts[stat.category];
      if (eventCount !== undefined && eventCount > 0) {
        return {
          ...stat,
          count: eventCount,
        };
      }
      return stat;
    });
  }, [categoryStats, events]);

  // Dynamically compute Events captured by Bus from real events
  const dynamicEventsByBus = useMemo(() => {
    const busMap: Record<string, number> = {};
    buses.slice(0, 7).forEach((b) => {
      busMap[b.id] = 0;
    });
    events.forEach((e) => {
      if (busMap[e.busId] !== undefined) {
        busMap[e.busId]++;
      } else {
        const normalized = e.busId.replace('BUS-0', 'BUS-');
        if (busMap[normalized] !== undefined) {
          busMap[normalized]++;
        }
      }
    });
    return Object.entries(busMap).map(([bus, count]) => ({
      bus,
      count,
    }));
  }, [buses, events]);

  // Exactly the 9 map layer filters
  const [filters, setFilters] = useState<MapLayerFilterState>({
    buses: true,
    roadDamage: true,
    traffic: true,
    vehicles: true,
    safety: true,
    accidents: true,
    waterlogging: true,
    infrastructure: true,
    anprOcr: true,
  });

  const toggleFilter = (key: keyof MapLayerFilterState) => {
    setFilters((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // Mini sparkline SVG generator for the 9 Event Intelligence cards
  const renderSparkline = (data: number[], isUp: boolean) => {
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    const width = 70;
    const height = 22;

    const points = data
      .map((val, idx) => {
        const x = (idx / (data.length - 1)) * width;
        const y = height - ((val - min) / range) * (height - 6) - 3;
        return `${x},${y}`;
      })
      .join(' ');

    const strokeColor = isUp ? '#15803D' : '#B91C1C';

    return (
      <svg width={width} height={height} className="overflow-visible">
        <polyline
          fill="none"
          stroke={strokeColor}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />
      </svg>
    );
  };

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
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* ================================================== */}
      {/* SECTION 1 — CITY MAP (FIRST MAJOR CONTENT SECTION) */}
      {/* ================================================== */}
      <section className="glass-panel p-6 rounded-2xl space-y-4">
        {/* Layer Filters Toolbar - EXACTLY the 9 filters required */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#46707E]"></span>
              <h3 className="text-sm font-extrabold uppercase tracking-wider text-[#22393C]">
                City Map & Geospatial Intelligence
              </h3>
            </div>
            <button
              onClick={() =>
                setFilters({
                  buses: true,
                  roadDamage: true,
                  traffic: true,
                  vehicles: true,
                  safety: true,
                  accidents: true,
                  waterlogging: true,
                  infrastructure: true,
                  anprOcr: true,
                })
              }
              className="text-xs font-semibold text-[#46707E] hover:text-[#22393C] underline"
            >
              Select All Filters
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-2 pt-1">
            {[
              { key: 'buses', label: 'Buses', icon: '🚌' },
              { key: 'roadDamage', label: 'Road Damage / Potholes', icon: '⚠️' },
              { key: 'traffic', label: 'Traffic & Congestion', icon: '🚦' },
              { key: 'vehicles', label: 'Vehicles', icon: '🚗' },
              { key: 'safety', label: 'Safety / Pedestrian Risk', icon: '🚸' },
              { key: 'accidents', label: 'Accidents / Incidents', icon: '💥' },
              { key: 'waterlogging', label: 'Waterlogging', icon: '🌊' },
              { key: 'infrastructure', label: 'Infrastructure Deficiencies', icon: '🏗️' },
              { key: 'anprOcr', label: 'ANPR / OCR Events', icon: '📷' },
            ].map((item) => {
              const active = filters[item.key as keyof MapLayerFilterState];
              return (
                <button
                  key={item.key}
                  data-filter-key={item.key}
                  onClick={() => toggleFilter(item.key as keyof MapLayerFilterState)}
                  className={`flex items-center space-x-2 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all border ${
                    active
                      ? 'bg-[#22393C] text-white border-[#22393C] shadow-sm'
                      : 'bg-white/70 text-gray-600 border-[#46707E]/20 hover:bg-white'
                  }`}
                >
                  <span className="text-xs">{item.icon}</span>
                  <span>{item.label}</span>
                  {active ? (
                    <CheckSquare className="w-3.5 h-3.5 text-[#AFBB98]" />
                  ) : (
                    <Square className="w-3.5 h-3.5 text-gray-400" />
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* City Map: Light Pune Map */}
        <div className="relative">
          <LightCityMap
            buses={buses}
            events={events}
            filters={filters}
            height="520px"
          />
        </div>

        {/* Map Legend */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 text-xs font-medium text-[#22393C]/80 border-t border-[#46707E]/15">
          <div className="flex items-center space-x-2">
            <span className="font-bold text-[#22393C]">Map Legend:</span>
            <span className="text-gray-500">Click any marker to inspect metadata or jump to Event Log</span>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs font-semibold">
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-600"></span>
              </span>
              Live Edge AI Detection
            </span>
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-gray-100 text-gray-700 border border-gray-200">
              <span className="w-2 h-2 rounded-full bg-gray-400 inline-block"></span>
              Prototype Baseline Context
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-[#22393C] border border-white inline-block"></span>
              Buses (PMPML)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-[#46707E] border border-white inline-block"></span>
              Road Damage / Potholes
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-[#546F67] border border-white inline-block"></span>
              Traffic Delay
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-[#93A17B] border border-white inline-block"></span>
              Safety Risk
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-600 border border-white inline-block"></span>
              Critical Accidents
            </span>
          </div>
        </div>
      </section>

      {/* ================================================== */}
      {/* SECTION 2 — EVENT INTELLIGENCE (9 MANDATORY CARDS) */}
      {/* ================================================== */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[#6B8B81]"></span>
            <h3 className="text-sm font-extrabold uppercase tracking-wider text-[#22393C]">
              Event Intelligence
            </h3>
          </div>
          <span className="text-xs text-[#46707E] font-medium">
            Centralized Real-Time Event Aggregates
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3.5">
          {dynamicCategoryStats.map((item) => (
            <div
              key={item.category}
              onClick={() => setCurrentModule('event-log')}
              className="glass-panel p-4 rounded-2xl cursor-pointer transition-all hover:shadow-glass-hover border border-[#46707E]/20 hover:border-[#46707E]/40"
              title="Click to view category in Event Log"
            >
              <div className="flex items-center justify-between">
                <span className="text-lg">{getCategoryIcon(item.category)}</span>
                <div
                  className={`flex items-center space-x-1 text-[11px] font-bold ${
                    item.isUp ? 'text-emerald-700' : 'text-rose-700'
                  }`}
                >
                  {item.isUp ? (
                    <TrendingUp className="w-3 h-3" />
                  ) : (
                    <TrendingDown className="w-3 h-3" />
                  )}
                  <span>{item.change}</span>
                </div>
              </div>

              <div className="mt-2.5">
                <div className="text-xs font-bold text-[#46707E] truncate">
                  {item.category}
                </div>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-2xl font-extrabold font-mono text-[#22393C]">
                    {item.count}
                  </span>
                  <div className="opacity-80">
                    {renderSparkline(item.sparkline, item.isUp)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ================================================== */}
      {/* SECTION 3 — INSIGHTS & ANALYTICS (OVERVIEW)        */}
      {/* ================================================== */}
      <section className="space-y-6">
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[#AFBB98]"></span>
          <h3 className="text-sm font-extrabold uppercase tracking-wider text-[#22393C]">
            Insights & Analytics Overview
          </h3>
        </div>

        {/* Charts Row 1: Events by Type & Event Trends */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Events by Type (Donut) */}
          <div className="lg:col-span-5 glass-panel p-5 rounded-2xl space-y-3">
            <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-2.5">
              <h4 className="text-xs font-extrabold uppercase tracking-wider text-[#22393C]">
                Events by Type
              </h4>
              <span className="text-xs font-mono font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                {events.length} Live AI Events
              </span>
            </div>

            <div className="h-60 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={dynamicEventsByType.length > 0 ? dynamicEventsByType : analyticsData.eventsByType}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={80}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {(dynamicEventsByType.length > 0 ? dynamicEventsByType : analyticsData.eventsByType).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: '0.75rem',
                      border: '1px solid rgba(70, 112, 126, 0.25)',
                      fontSize: '12px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[#46707E]/10 text-[11px]">
              {(dynamicEventsByType.length > 0 ? dynamicEventsByType : analyticsData.eventsByType).slice(0, 6).map((item) => (
                <div key={item.name} className="flex items-center space-x-1.5">
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: item.color }}
                  ></span>
                  <span className="truncate text-gray-700" title={item.name}>{item.name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Event Trends (Last 7 Days) */}
          <div className="lg:col-span-7 glass-panel p-5 rounded-2xl space-y-3">
            <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-2.5">
              <h4 className="text-xs font-extrabold uppercase tracking-wider text-[#22393C]">
                Event Trends (Last 7 Days)
              </h4>
              <span className="text-xs font-semibold text-emerald-700 flex items-center gap-1">
                <TrendingUp className="w-3.5 h-3.5" /> Stable Discovery Curve
              </span>
            </div>

            <div className="h-60 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={analyticsData.eventTrends7Days}>
                  <defs>
                    <linearGradient id="dashTotalGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#46707E" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#46707E" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="dashRoadGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#AFBB98" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#AFBB98" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#46707E" opacity={0.15} />
                  <XAxis dataKey="day" stroke="#46707E" fontSize={11} />
                  <YAxis stroke="#46707E" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: '0.75rem',
                      border: '1px solid rgba(70, 112, 126, 0.25)',
                      fontSize: '12px',
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="Total"
                    stroke="#46707E"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#dashTotalGrad)"
                  />
                  <Area
                    type="monotone"
                    dataKey="RoadIssues"
                    stroke="#6B8B81"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#dashRoadGrad)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="flex items-center justify-center space-x-6 text-xs text-gray-600 pt-1">
              <span className="flex items-center gap-2">
                <span className="w-3 h-1 bg-[#46707E] rounded-full inline-block"></span> Total Events
              </span>
              <span className="flex items-center gap-2">
                <span className="w-3 h-1 bg-[#6B8B81] rounded-full inline-block"></span> Road Issues
              </span>
            </div>
          </div>
        </div>

        {/* Charts Row 2: Road Issues by Area & Traffic Trends */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Road Issues by Area */}
          <div className="glass-panel p-5 rounded-2xl space-y-3">
            <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-2.5">
              <h4 className="text-xs font-extrabold uppercase tracking-wider text-[#22393C]">
                Road Issues by Area (Pune Zones)
              </h4>
              <span className="text-xs text-gray-500 font-mono">Potholes & Deficiencies</span>
            </div>

            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analyticsData.roadIssuesByArea}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#46707E" opacity={0.15} />
                  <XAxis dataKey="area" stroke="#46707E" fontSize={11} />
                  <YAxis stroke="#46707E" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: '0.75rem',
                      border: '1px solid rgba(70, 112, 126, 0.25)',
                      fontSize: '12px',
                    }}
                  />
                  <Bar dataKey="count" fill="#46707E" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Traffic Trends */}
          <div className="glass-panel p-5 rounded-2xl space-y-3">
            <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-2.5">
              <h4 className="text-xs font-extrabold uppercase tracking-wider text-[#22393C]">
                Traffic Trends (Congestion Index by Hour)
              </h4>
              <span className="text-xs text-rose-700 font-bold font-mono">Peak Spikes @ 09:00 & 18:00</span>
            </div>

            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={analyticsData.trafficTrends}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#46707E" opacity={0.15} />
                  <XAxis dataKey="hour" stroke="#46707E" fontSize={11} />
                  <YAxis stroke="#46707E" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: '0.75rem',
                      border: '1px solid rgba(70, 112, 126, 0.25)',
                      fontSize: '12px',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="index"
                    stroke="#B91C1C"
                    strokeWidth={2.5}
                    dot={{ fill: '#B91C1C', r: 3 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Row 3: Events by Bus */}
        <div className="glass-panel p-5 rounded-2xl space-y-3">
          <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-2.5">
            <h4 className="text-xs font-extrabold uppercase tracking-wider text-[#22393C]">
              Events Captured by Bus (Edge AI Discovery Distribution)
            </h4>
            <span className="text-xs text-emerald-700 font-mono font-bold">Live Telemetry Yield</span>
          </div>

          <div className="h-52 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dynamicEventsByBus.length > 0 ? dynamicEventsByBus : analyticsData.eventsByBus}>
                <CartesianGrid strokeDasharray="3 3" stroke="#46707E" opacity={0.15} />
                <XAxis dataKey="bus" stroke="#46707E" fontSize={11} />
                <YAxis stroke="#46707E" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    borderRadius: '0.75rem',
                    border: '1px solid rgba(70, 112, 126, 0.25)',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="count" fill="#6B8B81" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* High Severity Locations Ranking Table */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-3">
            <div className="flex items-center space-x-2">
              <Flame className="w-4 h-4 text-rose-600" />
              <h4 className="text-sm font-extrabold text-[#22393C] uppercase tracking-wide">
                High Severity Incident Locations
              </h4>
            </div>
            <span className="text-xs text-gray-500 font-mono">Ranked by Risk Weight</span>
          </div>

          <div className="overflow-x-auto rounded-xl border border-[#46707E]/20">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#22393C]/5 text-[#46707E] font-bold uppercase tracking-wider border-b border-[#46707E]/15">
                <tr>
                  <th className="py-3 px-4">Location Corridor</th>
                  <th className="py-3 px-4">Severity Level</th>
                  <th className="py-3 px-4">Total Defects</th>
                  <th className="py-3 px-4 text-right">Transit Impact</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#46707E]/10 bg-white/40">
                {analyticsData.highSeverityLocations.map((loc, idx) => (
                  <tr key={idx} className="hover:bg-white/70 transition-colors">
                    <td className="py-3 px-4 font-bold text-[#22393C]">
                      <div className="flex items-center space-x-2">
                        <span className="w-5 h-5 rounded-full bg-black/5 text-[#46707E] font-mono font-bold flex items-center justify-center text-[10px]">
                          {idx + 1}
                        </span>
                        <span>{loc.location}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          loc.severity === 'CRITICAL'
                            ? 'bg-rose-100 text-rose-800'
                            : 'bg-amber-100 text-amber-800'
                        }`}
                      >
                        {loc.severity}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono font-semibold text-[#22393C]">
                      {loc.issues} issues flagged
                    </td>
                    <td className="py-3 px-4 font-mono font-bold text-rose-700 text-right">
                      {loc.delay}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Compact Origin-Destination Traffic Patterns Section */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-3">
            <div className="flex items-center space-x-2">
              <Compass className="w-4 h-4 text-[#46707E]" />
              <h4 className="text-sm font-extrabold text-[#22393C] uppercase tracking-wide">
                Traffic Flow Patterns
              </h4>
            </div>
            <span className="text-xs text-gray-500 font-mono">
              Origin → Destination Analysis
            </span>
          </div>

          <div className="overflow-x-auto rounded-xl border border-[#46707E]/20">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#22393C]/5 text-[#46707E] font-bold uppercase tracking-wider border-b border-[#46707E]/15">
                <tr>
                  <th className="py-3 px-4">Origin</th>
                  <th className="py-3 px-4">Destination</th>
                  <th className="py-3 px-4">Peak Flow</th>
                  <th className="py-3 px-4 text-right">Avg. Delay</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#46707E]/10 bg-white/40">
                {analyticsData.originDestinationPatterns.map((row, idx) => (
                  <tr key={idx} className="hover:bg-white/70 transition-colors">
                    <td className="py-3 px-4 font-bold text-[#22393C]">{row.origin}</td>
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-1.5 font-bold text-[#46707E]">
                        <ArrowRight className="w-3.5 h-3.5 text-gray-400" />
                        <span>{row.destination}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold ${
                          row.peakFlow === 'High'
                            ? 'bg-rose-100 text-rose-800'
                            : 'bg-blue-100 text-blue-800'
                        }`}
                      >
                        {row.peakFlow}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono font-bold text-rose-700 text-right">
                      {row.avgDelay}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Key Insights Section (3 Cards) */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center space-x-2 border-b border-[#46707E]/20 pb-3">
            <Lightbulb className="w-4 h-4 text-[#AFBB98]" />
            <h4 className="text-sm font-extrabold text-[#22393C] uppercase tracking-wide">
              Key Insights
            </h4>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {analyticsData.keyInsights.map((insight, idx) => (
              <div
                key={idx}
                className="p-4 rounded-xl bg-white/75 border border-[#46707E]/20 space-y-2 hover:border-[#46707E]/40 transition-all"
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-[#46707E]/15 text-[#46707E]">
                    {insight.tag}
                  </span>
                  <span className="text-xs font-mono font-bold text-gray-400">0{idx + 1}</span>
                </div>
                <h5 className="text-xs font-bold text-[#22393C]">{insight.title}</h5>
                <p className="text-xs text-gray-600 leading-relaxed">{insight.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

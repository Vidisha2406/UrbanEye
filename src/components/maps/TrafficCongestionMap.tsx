import React, { useState, useMemo } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { useUrbanEye } from '../../context/UrbanEyeContext';

type TimeRangeKey = 'live' | 'morning' | 'afternoon' | 'evening';

interface CorridorData {
  id: string;
  name: string;
  coords: [number, number][];
  // Severity per time range: 'High' (Red), 'Medium' (Amber), 'Low' (Green)
  severities: Record<TimeRangeKey, 'High' | 'Medium' | 'Low'>;
  delays: Record<TimeRangeKey, string>;
  speeds: Record<TimeRangeKey, string>;
  persistence: string;
  liveTelemetry?: {
    vehicleCount: number;
    trafficDensity: string;
    congestionLevel: string;
    vehicleMix: Record<string, string>;
  };
}

interface BottleneckNode {
  id: string;
  name: string;
  coords: [number, number];
  delays: Record<TimeRangeKey, string>;
  severities: Record<TimeRangeKey, 'High' | 'Medium' | 'Low'>;
  flowSpeed: Record<TimeRangeKey, string>;
}

// 12 Real Pune Arterial Traffic Corridors with dense coordinate polylines
const puneCorridors: CorridorData[] = [
  {
    id: 'CORR-FC',
    name: 'FC Road (Fergusson College Road)',
    coords: [
      [18.5308, 73.8475],
      [18.5280, 73.8440],
      [18.5240, 73.8425],
      [18.5218, 73.8415],
      [18.5175, 73.8420],
      [18.5145, 73.8435],
    ],
    severities: {
      live: 'High',
      morning: 'High',
      afternoon: 'Medium',
      evening: 'High',
    },
    delays: {
      live: '+18 min',
      morning: '+22 min',
      afternoon: '+8 min',
      evening: '+25 min',
    },
    speeds: {
      live: '8 km/hr',
      morning: '6 km/hr',
      afternoon: '18 km/hr',
      evening: '5 km/hr',
    },
    persistence: '35 min tailback queue',
  },
  {
    id: 'CORR-UNIV',
    name: 'University Road (Ganeshkhind Road)',
    coords: [
      [18.5520, 73.8240],
      [18.5440, 73.8270],
      [18.5385, 73.8290],
      [18.5340, 73.8305],
      [18.5314, 73.8315],
      [18.5280, 73.8400],
      [18.5308, 73.8475],
    ],
    severities: {
      live: 'High',
      morning: 'High',
      afternoon: 'Medium',
      evening: 'High',
    },
    delays: {
      live: '+14 min',
      morning: '+20 min',
      afternoon: '+6 min',
      evening: '+19 min',
    },
    speeds: {
      live: '9 km/hr',
      morning: '7 km/hr',
      afternoon: '22 km/hr',
      evening: '8 km/hr',
    },
    persistence: '28 min tailback queue',
  },
  {
    id: 'CORR-JM',
    name: 'JM Road (Jangali Maharaj Road)',
    coords: [
      [18.5308, 73.8475],
      [18.5280, 73.8480],
      [18.5246, 73.8488],
      [18.5200, 73.8480],
      [18.5165, 73.8460],
      [18.5135, 73.8440],
    ],
    severities: {
      live: 'Medium',
      morning: 'High',
      afternoon: 'Medium',
      evening: 'High',
    },
    delays: {
      live: '+11 min',
      morning: '+15 min',
      afternoon: '+5 min',
      evening: '+18 min',
    },
    speeds: {
      live: '14 km/hr',
      morning: '11 km/hr',
      afternoon: '24 km/hr',
      evening: '10 km/hr',
    },
    persistence: '22 min queue',
  },
  {
    id: 'CORR-KHARADI',
    name: 'Kharadi Bypass (Mundhwa - Kharadi IT Corridor)',
    coords: [
      [18.5400, 73.8950],
      [18.5463, 73.9034],
      [18.5500, 73.9200],
      [18.5529, 73.9352],
      [18.5580, 73.9500],
    ],
    severities: {
      live: 'Medium',
      morning: 'High',
      afternoon: 'Low',
      evening: 'High',
    },
    delays: {
      live: '+7 min',
      morning: '+14 min',
      afternoon: '+2 min',
      evening: '+16 min',
    },
    speeds: {
      live: '19 km/hr',
      morning: '12 km/hr',
      afternoon: '36 km/hr',
      evening: '11 km/hr',
    },
    persistence: '15 min queue',
  },
  {
    id: 'CORR-AUNDH',
    name: 'Aundh - Baner Link Road',
    coords: [
      [18.5650, 73.8150],
      [18.5601, 73.8073],
      [18.5580, 73.7980],
      [18.5590, 73.7868],
      [18.5620, 73.7750],
    ],
    severities: {
      live: 'Low',
      morning: 'Medium',
      afternoon: 'Low',
      evening: 'Medium',
    },
    delays: {
      live: '+2 min',
      morning: '+8 min',
      afternoon: '+1 min',
      evening: '+9 min',
    },
    speeds: {
      live: '38 km/hr',
      morning: '22 km/hr',
      afternoon: '42 km/hr',
      evening: '20 km/hr',
    },
    persistence: 'Free Flowing Arterial',
  },
  {
    id: 'CORR-SINHAGAD',
    name: 'Sinhagad Road (Dandekar Bridge - Manikbaug)',
    coords: [
      [18.5080, 73.8450],
      [18.5018, 73.8400],
      [18.4950, 73.8350],
      [18.4900, 73.8300],
      [18.4820, 73.8240],
      [18.4750, 73.8180],
    ],
    severities: {
      live: 'High',
      morning: 'High',
      afternoon: 'Medium',
      evening: 'High',
    },
    delays: {
      live: '+12 min',
      morning: '+16 min',
      afternoon: '+6 min',
      evening: '+24 min',
    },
    speeds: {
      live: '11 km/hr',
      morning: '9 km/hr',
      afternoon: '24 km/hr',
      evening: '7 km/hr',
    },
    persistence: '20 min tailback queue',
  },
  {
    id: 'CORR-SWARGATE',
    name: 'Swargate - Shivaji Road Transit Corridor',
    coords: [
      [18.5246, 73.8560],
      [18.5180, 73.8570],
      [18.5100, 73.8575],
      [18.5018, 73.8580],
      [18.4950, 73.8590],
    ],
    severities: {
      live: 'High',
      morning: 'High',
      afternoon: 'High',
      evening: 'High',
    },
    delays: {
      live: '+16 min',
      morning: '+19 min',
      afternoon: '+12 min',
      evening: '+22 min',
    },
    speeds: {
      live: '7 km/hr',
      morning: '6 km/hr',
      afternoon: '11 km/hr',
      evening: '5 km/hr',
    },
    persistence: 'Heavy Transit Congestion',
  },
  {
    id: 'CORR-KARVE',
    name: 'Karve Road (Deccan Gymkhana to Kothrud Stand)',
    coords: [
      [18.5145, 73.8435],
      [18.5110, 73.8350],
      [18.5090, 73.8240],
      [18.5074, 73.8077],
      [18.5040, 73.7950],
    ],
    severities: {
      live: 'Medium',
      morning: 'High',
      afternoon: 'Low',
      evening: 'High',
    },
    delays: {
      live: '+8 min',
      morning: '+14 min',
      afternoon: '+3 min',
      evening: '+16 min',
    },
    speeds: {
      live: '18 km/hr',
      morning: '11 km/hr',
      afternoon: '34 km/hr',
      evening: '10 km/hr',
    },
    persistence: '18 min queue',
  },
  {
    id: 'CORR-HADAPSAR',
    name: 'Pune-Solapur Road (Swargate to Hadapsar Gadital)',
    coords: [
      [18.5018, 73.8580],
      [18.5025, 73.8750],
      [18.5030, 73.8950],
      [18.5025, 73.9150],
      [18.5020, 73.9280],
      [18.5010, 73.9450],
    ],
    severities: {
      live: 'High',
      morning: 'High',
      afternoon: 'Medium',
      evening: 'High',
    },
    delays: {
      live: '+15 min',
      morning: '+21 min',
      afternoon: '+7 min',
      evening: '+26 min',
    },
    speeds: {
      live: '9 km/hr',
      morning: '7 km/hr',
      afternoon: '21 km/hr',
      evening: '6 km/hr',
    },
    persistence: 'Flyover Ramp Constriction',
  },
  {
    id: 'CORR-SB',
    name: 'Senapati Bapat Road',
    coords: [
      [18.5385, 73.8290],
      [18.5320, 73.8300],
      [18.5250, 73.8320],
      [18.5180, 73.8340],
    ],
    severities: {
      live: 'Medium',
      morning: 'Medium',
      afternoon: 'Low',
      evening: 'Medium',
    },
    delays: {
      live: '+5 min',
      morning: '+9 min',
      afternoon: '+2 min',
      evening: '+10 min',
    },
    speeds: {
      live: '24 km/hr',
      morning: '18 km/hr',
      afternoon: '35 km/hr',
      evening: '16 km/hr',
    },
    persistence: '12 min queue',
  },
  {
    id: 'CORR-VIMAN',
    name: 'Ahmednagar Road (Yerwada to Viman Nagar)',
    coords: [
      [18.5520, 73.8820],
      [18.5550, 73.8950],
      [18.5610, 73.9050],
      [18.5679, 73.9143],
      [18.5720, 73.9300],
    ],
    severities: {
      live: 'Medium',
      morning: 'High',
      afternoon: 'Low',
      evening: 'High',
    },
    delays: {
      live: '+8 min',
      morning: '+15 min',
      afternoon: '+3 min',
      evening: '+18 min',
    },
    speeds: {
      live: '21 km/hr',
      morning: '12 km/hr',
      afternoon: '38 km/hr',
      evening: '11 km/hr',
    },
    persistence: 'Bus Rapid Transit Lane Spillover',
  },
  {
    id: 'CORR-PASHAN',
    name: 'Pashan - Baner Outer Link',
    coords: [
      [18.5420, 73.7920],
      [18.5480, 73.7880],
      [18.5540, 73.7840],
      [18.5590, 73.7868],
    ],
    severities: {
      live: 'Low',
      morning: 'Low',
      afternoon: 'Low',
      evening: 'Low',
    },
    delays: {
      live: '+1 min',
      morning: '+3 min',
      afternoon: '+0 min',
      evening: '+4 min',
    },
    speeds: {
      live: '44 km/hr',
      morning: '38 km/hr',
      afternoon: '48 km/hr',
      evening: '36 km/hr',
    },
    persistence: 'Uncongested Free Flow',
  },
];

// Key Traffic Bottleneck Intersection Chokepoints
const bottleneckNodes: BottleneckNode[] = [
  {
    id: 'BN-01',
    name: 'University Circle / E-Square Chokepoint',
    coords: [18.5314, 73.8315],
    severities: { live: 'High', morning: 'High', afternoon: 'Medium', evening: 'High' },
    delays: { live: '+14 min', morning: '+20 min', afternoon: '+6 min', evening: '+19 min' },
    flowSpeed: { live: '8 km/hr', morning: '6 km/hr', afternoon: '19 km/hr', evening: '7 km/hr' },
  },
  {
    id: 'BN-02',
    name: 'Deccan Gymkhana / Goodluck Chowk',
    coords: [18.5218, 73.8415],
    severities: { live: 'High', morning: 'High', afternoon: 'Medium', evening: 'High' },
    delays: { live: '+18 min', morning: '+22 min', afternoon: '+7 min', evening: '+25 min' },
    flowSpeed: { live: '6 km/hr', morning: '5 km/hr', afternoon: '18 km/hr', evening: '5 km/hr' },
  },
  {
    id: 'BN-03',
    name: 'Sancheti Hospital Chowk / Flyover Merge',
    coords: [18.5308, 73.8475],
    severities: { live: 'High', morning: 'High', afternoon: 'Medium', evening: 'High' },
    delays: { live: '+11 min', morning: '+15 min', afternoon: '+5 min', evening: '+17 min' },
    flowSpeed: { live: '9 km/hr', morning: '8 km/hr', afternoon: '22 km/hr', evening: '8 km/hr' },
  },
  {
    id: 'BN-04',
    name: 'Swargate Multi-Modal Transit Hub',
    coords: [18.5018, 73.8580],
    severities: { live: 'High', morning: 'High', afternoon: 'High', evening: 'High' },
    delays: { live: '+16 min', morning: '+19 min', afternoon: '+12 min', evening: '+22 min' },
    flowSpeed: { live: '7 km/hr', morning: '5 km/hr', afternoon: '11 km/hr', evening: '6 km/hr' },
  },
  {
    id: 'BN-05',
    name: 'Hadapsar Gadital Junction',
    coords: [18.5020, 73.9280],
    severities: { live: 'High', morning: 'High', afternoon: 'Medium', evening: 'High' },
    delays: { live: '+15 min', morning: '+21 min', afternoon: '+7 min', evening: '+26 min' },
    flowSpeed: { live: '9 km/hr', morning: '6 km/hr', afternoon: '20 km/hr', evening: '6 km/hr' },
  },
  {
    id: 'BN-06',
    name: 'Chandani Chowk Highway Interchange',
    coords: [18.5040, 73.7850],
    severities: { live: 'Medium', morning: 'High', afternoon: 'Low', evening: 'High' },
    delays: { live: '+5 min', morning: '+11 min', afternoon: '+2 min', evening: '+12 min' },
    flowSpeed: { live: '26 km/hr', morning: '16 km/hr', afternoon: '40 km/hr', evening: '15 km/hr' },
  },
];

// Helper to create traffic bottleneck radar marker
const createBottleneckIcon = (severity: 'High' | 'Medium' | 'Low', delay: string) => {
  const color = severity === 'High' ? '#DC2626' : severity === 'Medium' ? '#D97706' : '#15803D';
  const pulseClass = severity === 'High' ? 'animate-ping' : '';

  return L.divIcon({
    className: 'traffic-bottleneck-marker',
    html: `
      <div style="position: relative; display: flex; align-items: center; justify-content: center;">
        <!-- Pulsing radar glow -->
        <div class="${pulseClass}" style="
          position: absolute;
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: ${color};
          opacity: 0.35;
        "></div>
        <!-- Center icon button -->
        <div style="
          position: relative;
          width: 26px;
          height: 26px;
          border-radius: 50%;
          background: #22393C;
          border: 2.5px solid ${color};
          box-shadow: 0 4px 10px rgba(0,0,0,0.35);
          display: flex;
          align-items: center;
          justify-content: center;
          color: #ffffff;
          font-size: 11px;
          font-weight: 800;
        ">
          ⚠️
        </div>
        <!-- Delay pill tag -->
        <div style="
          position: absolute;
          top: -18px;
          white-space: nowrap;
          background: #22393C;
          color: ${color};
          border: 1px solid ${color};
          font-size: 9px;
          font-family: monospace;
          font-weight: 800;
          padding: 1px 4px;
          border-radius: 4px;
          box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        ">
          ${delay}
        </div>
      </div>
    `,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -18],
  });
};

export const TrafficCongestionMap: React.FC = () => {
  const { latestTrafficTelemetry, events } = useUrbanEye();
  const [timeRange, setTimeRange] = useState<TimeRangeKey>('live');

  // Corridor location keywords for spatial matching
  const corridorKeywords: Record<string, string[]> = {
    'CORR-FC': ['fc road', 'fergusson'],
    'CORR-UNIV': ['university', 'ganeshkhind'],
    'CORR-JM': ['jm road', 'jangali maharaj'],
    'CORR-KHARADI': ['kharadi', 'mundhwa'],
    'CORR-AUNDH': ['aundh', 'baner'],
    'CORR-SINHAGAD': ['sinhagad', 'dandekar', 'manikbaug'],
    'CORR-SWARGATE': ['swargate', 'shivaji road'],
    'CORR-KARVE': ['karve', 'kothrud', 'deccan'],
    'CORR-HADAPSAR': ['hadapsar', 'gadital', 'solapur road'],
    'CORR-SB': ['senapati bapat', 'sb road'],
    'CORR-VIMAN': ['viman', 'viman nagar', 'ahmednagar', 'yerwada'],
    'CORR-PASHAN': ['pashan'],
  };

  // Color mapping: Green = Free Flow, Amber = Moderate, Red = High/Severe
  const getCorridorColor = (severity: 'High' | 'Medium' | 'Low') => {
    switch (severity) {
      case 'High':
        return '#DC2626'; // Vivid Red (Severe/High Congestion)
      case 'Medium':
        return '#F59E0B'; // Rich Amber/Yellow (Moderate Congestion)
      case 'Low':
        return '#16A34A'; // Bright Green (Free Flow)
    }
  };

  const timeRangeLabels: { id: TimeRangeKey; label: string; desc: string }[] = [
    { id: 'live', label: 'Live Now', desc: 'Real-time telemetry' },
    { id: 'morning', label: 'Morning Peak (08:30 - 10:30)', desc: 'Commuter inflow' },
    { id: 'afternoon', label: 'Afternoon (13:00 - 15:00)', desc: 'Off-peak transit' },
    { id: 'evening', label: 'Evening Peak (17:30 - 20:00)', desc: 'Return gridlock' },
  ];

  // Determine target corridor matched by latest real traffic event
  const matchedCorridorId = useMemo(() => {
    const latestTrafficEvent = events.find(
      (e) => e.type === 'TRAFFIC_CONGESTION' || e.category === 'Traffic'
    );
    if (!latestTrafficEvent) return 'CORR-VIMAN';
    const loc = (latestTrafficEvent.locationName || '').toLowerCase();
    for (const [corrId, kws] of Object.entries(corridorKeywords)) {
      if (kws.some((kw) => loc.includes(kw))) {
        return corrId;
      }
    }
    return 'CORR-VIMAN';
  }, [events]);

  // Dynamically update corridors when live AI telemetry is received
  const activeCorridors = useMemo(() => {
    if (!latestTrafficTelemetry) return puneCorridors;

    const level = latestTrafficTelemetry.congestionLevel;
    let sev: 'High' | 'Medium' | 'Low' = 'High';
    let delay = '+18 min';
    let speed = '12 km/hr';

    if (level === 'FREE FLOW' || level === 'LOW') {
      sev = 'Low';
      delay = '+1 min';
      speed = '40 km/hr';
    } else if (level === 'MODERATE') {
      sev = 'Medium';
      delay = '+8 min';
      speed = '22 km/hr';
    } else if (level === 'HIGH') {
      sev = 'High';
      delay = '+18 min';
      speed = '12 km/hr';
    } else if (level === 'SEVERE') {
      sev = 'High';
      delay = '+28 min';
      speed = '7 km/hr';
    }

    return puneCorridors.map((c) => {
      if (c.id === matchedCorridorId || (matchedCorridorId === 'CORR-VIMAN' && c.id === 'CORR-VIMAN')) {
        return {
          ...c,
          severities: {
            ...c.severities,
            live: sev,
          },
          delays: {
            ...c.delays,
            live: delay,
          },
          speeds: {
            ...c.speeds,
            live: speed,
          },
          persistence: `Live AI Inference: ${latestTrafficTelemetry.vehicleCount} vehicles (${latestTrafficTelemetry.trafficDensity} Density, ${latestTrafficTelemetry.congestionLevel})`,
          liveTelemetry: {
            vehicleCount: latestTrafficTelemetry.vehicleCount,
            trafficDensity: latestTrafficTelemetry.trafficDensity,
            congestionLevel: latestTrafficTelemetry.congestionLevel,
            vehicleMix: latestTrafficTelemetry.vehicleMix,
          },
        };
      }
      return c;
    });
  }, [latestTrafficTelemetry, matchedCorridorId]);

  // Dynamically inject live AI verified bottleneck at traffic coordinates
  const activeBottlenecks = useMemo(() => {
    if (!latestTrafficTelemetry) return bottleneckNodes;

    const latestTrafficEvent = events.find(
      (e) => e.type === 'TRAFFIC_CONGESTION' || e.category === 'Traffic'
    );

    const level = latestTrafficTelemetry.congestionLevel;
    const sev: 'High' | 'Medium' | 'Low' =
      level === 'SEVERE' || level === 'HIGH' ? 'High' : level === 'MODERATE' ? 'Medium' : 'Low';
    const delay =
      level === 'SEVERE' ? '+28 min' : level === 'HIGH' ? '+18 min' : level === 'MODERATE' ? '+8 min' : '+1 min';
    const speed =
      level === 'SEVERE' ? '7 km/hr' : level === 'HIGH' ? '12 km/hr' : level === 'MODERATE' ? '22 km/hr' : '40 km/hr';

    const coords: [number, number] = latestTrafficEvent?.coordinates || [18.5679, 73.9143];
    const locName = latestTrafficEvent?.locationName || 'Ahmednagar Rd / Viman Nagar';

    const aiNode: BottleneckNode = {
      id: 'BN-AI-LIVE',
      name: `${locName} (AI Verified)`,
      coords,
      severities: {
        live: sev,
        morning: 'High',
        afternoon: 'Low',
        evening: 'High',
      },
      delays: {
        live: delay,
        morning: '+15 min',
        afternoon: '+3 min',
        evening: '+18 min',
      },
      flowSpeed: {
        live: speed,
        morning: '12 km/hr',
        afternoon: '38 km/hr',
        evening: '11 km/hr',
      },
    };
    return [aiNode, ...bottleneckNodes.filter((b) => b.id !== 'BN-AI-LIVE')];
  }, [latestTrafficTelemetry, events]);

  // Active corridor display name for the banner
  const activeCorridorName = useMemo(() => {
    const found = puneCorridors.find((c) => c.id === matchedCorridorId);
    return found ? found.name : 'Ahmednagar Road (Viman Nagar)';
  }, [matchedCorridorId]);

  // Dynamic summary stats for the current time range
  const summaryStats = useMemo(() => {
    const highCount = activeCorridors.filter((c) => c.severities[timeRange] === 'High').length;
    const medCount = activeCorridors.filter((c) => c.severities[timeRange] === 'Medium').length;
    const lowCount = activeCorridors.filter((c) => c.severities[timeRange] === 'Low').length;

    let avgSpeed = 14;
    if (timeRange === 'morning') avgSpeed = 10;
    if (timeRange === 'afternoon') avgSpeed = 28;
    if (timeRange === 'evening') avgSpeed = 8;
    if (timeRange === 'live' && latestTrafficTelemetry) {
      avgSpeed =
        latestTrafficTelemetry.congestionLevel === 'SEVERE'
          ? 7
          : latestTrafficTelemetry.congestionLevel === 'HIGH'
          ? 11
          : latestTrafficTelemetry.congestionLevel === 'MODERATE'
          ? 20
          : 32;
    }

    return { highCount, medCount, lowCount, avgSpeed };
  }, [timeRange, activeCorridors, latestTrafficTelemetry]);

  return (
    <div className="space-y-4">
      {/* Live Edge AI Telemetry Notification Banner */}
      {latestTrafficTelemetry && (
        <div className="flex items-center justify-between p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-900 text-xs font-medium animate-fadeIn">
          <div className="flex items-center space-x-2">
            <span className="flex h-2.5 w-2.5 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-600"></span>
            </span>
            <span>
              <strong>LIVE EDGE AI TELEMETRY ACTIVE:</strong> {activeCorridorName} corridor updated from real photo inference —{' '}
              <span className="font-mono font-bold text-emerald-800">{latestTrafficTelemetry.vehicleCount} vehicles detected</span> ({latestTrafficTelemetry.trafficDensity} Density, {latestTrafficTelemetry.congestionLevel} Congestion).
            </span>
          </div>
          <span className="text-[11px] font-mono text-emerald-700 bg-emerald-100/60 px-2.5 py-0.5 rounded border border-emerald-200 font-bold">
            YOLO11n Real Inference
          </span>
        </div>
      )}

      {/* Top Map Toolbar: Time Range Selector & Live Traffic Velocity Metrics */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white/80 backdrop-blur-md p-3.5 rounded-xl border border-[#46707E]/20">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-[#46707E]">
            Time Simulation:
          </span>
          <div className="flex flex-wrap bg-[#22393C]/10 p-0.5 rounded-lg border border-[#46707E]/20">
            {timeRangeLabels.map((t) => (
              <button
                key={t.id}
                onClick={() => setTimeRange(t.id)}
                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all ${
                  timeRange === t.id
                    ? 'bg-[#22393C] text-white shadow-sm'
                    : 'text-[#22393C] hover:text-[#46707E] hover:bg-white/40'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {/* Dynamic Speed & Chokepoints Pill */}
        <div className="flex items-center space-x-3 text-xs font-mono">
          <div className="px-3 py-1 rounded-lg bg-black/5 text-[#22393C] font-semibold border border-black/10">
            Avg Velocity: <strong className="text-[#46707E]">{summaryStats.avgSpeed} km/hr</strong>
          </div>
          <div className="px-3 py-1 rounded-lg bg-rose-50 text-rose-800 font-bold border border-rose-200">
            {summaryStats.highCount} Heavy Corridors
          </div>
        </div>
      </div>

      {/* Interactive Light Traffic Congestion Map */}
      <div className="w-full rounded-2xl overflow-hidden shadow-glass border border-[#46707E]/25 relative z-10">
        <MapContainer
          center={[18.5250, 73.8500]}
          zoom={12}
          style={{ height: '520px', width: '100%' }}
          scrollWheelZoom={false}
        >
          {/* CARTO Positron Light Basemap with authenticated key */}
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions" target="_blank" rel="noreferrer">CARTO</a>'
            url={`https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png${
              import.meta.env.VITE_CARTO_API_KEY
                ? `?key=${import.meta.env.VITE_CARTO_API_KEY}`
                : ''
            }`}
            subdomains="abcd"
            maxZoom={19}
          />

          {/* ROAD CORRIDOR TRAFFIC FLOW VISUALIZATION */}
          {/* Multi-layered polylines: outer glow + core flow line */}
          {activeCorridors.map((corridor) => {
            const currentSeverity = corridor.severities[timeRange];
            const color = getCorridorColor(currentSeverity);
            const delay = corridor.delays[timeRange];
            const speed = corridor.speeds[timeRange];

            return (
              <React.Fragment key={corridor.id}>
                {/* 1. Outer translucent buffer / road glow */}
                <Polyline
                  positions={corridor.coords}
                  pathOptions={{
                    color: color,
                    weight: 12,
                    opacity: 0.35,
                    lineCap: 'round',
                    lineJoin: 'round',
                  }}
                />

                {/* 2. Inner crisp traffic core polyline */}
                <Polyline
                  positions={corridor.coords}
                  pathOptions={{
                    color: color,
                    weight: 6,
                    opacity: 0.95,
                    lineCap: 'round',
                    lineJoin: 'round',
                  }}
                >
                  <Popup>
                    <div className="p-3.5 min-w-[240px]">
                      <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-2 mb-2">
                        <span className="font-bold text-xs text-[#22393C]">{corridor.name}</span>
                        <span
                          className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                            currentSeverity === 'High'
                              ? 'bg-rose-100 text-rose-800 border border-rose-300'
                              : currentSeverity === 'Medium'
                              ? 'bg-amber-100 text-amber-800 border border-amber-300'
                              : 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                          }`}
                        >
                          {currentSeverity === 'High'
                            ? 'Severe Congestion'
                            : currentSeverity === 'Medium'
                            ? 'Moderate Slowdown'
                            : 'Free Flow'}
                        </span>
                      </div>

                      <div className="space-y-1.5 text-xs text-[#22393C]/80">
                        <div className="flex justify-between">
                          <span className="text-gray-500">Current Velocity:</span>
                          <span className="font-mono font-bold text-[#22393C]">{speed}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">Transit Delay:</span>
                          <span className="font-mono font-bold text-rose-600">{delay}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">Queue Persistence:</span>
                          <span className="font-medium text-[#22393C]">{corridor.persistence}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">Time Window:</span>
                          <span className="font-semibold text-[#46707E]">
                            {timeRange.toUpperCase()}
                          </span>
                        </div>
                        {corridor.liveTelemetry && timeRange === 'live' && (
                          <div className="pt-2 mt-2 border-t border-[#46707E]/20 space-y-1.5 text-xs">
                            <div className="flex justify-between">
                              <span className="text-gray-500">YOLO11n Vehicle Count:</span>
                              <span className="font-mono font-bold text-emerald-700">
                                {corridor.liveTelemetry.vehicleCount} ({corridor.liveTelemetry.trafficDensity} Density)
                              </span>
                            </div>
                            {corridor.liveTelemetry.vehicleMix && Object.keys(corridor.liveTelemetry.vehicleMix).length > 0 && (
                              <div className="flex justify-between">
                                <span className="text-gray-500">Vehicle Mix:</span>
                                <span className="font-mono font-medium text-[11px] text-[#22393C]">
                                  {Object.entries(corridor.liveTelemetry.vehicleMix)
                                    .map(([cls, pct]) => `${cls}: ${pct}`)
                                    .join(', ')}
                                </span>
                              </div>
                            )}
                            <div className="flex justify-between items-center pt-0.5">
                              <span className="text-gray-500">Telemetry Status:</span>
                              <span className="text-[10px] font-bold text-emerald-800 bg-emerald-100 px-2 py-0.5 rounded border border-emerald-300">
                                Live Edge AI Inferred
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </Popup>
                </Polyline>
              </React.Fragment>
            );
          })}

          {/* BOTTLENECK CHOKEPOINT INDICATORS */}
          {activeBottlenecks.map((node) => {
            const severity = node.severities[timeRange];
            const delay = node.delays[timeRange];
            const speed = node.flowSpeed[timeRange];
            const isAiNode = node.id.startsWith('BN-AI');

            return (
              <Marker
                key={node.id}
                position={node.coords}
                icon={createBottleneckIcon(severity, delay)}
              >
                <Popup>
                  <div className="p-3.5 min-w-[240px]">
                    <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-2 mb-2">
                      <div>
                        <h4 className="font-bold text-xs text-[#22393C]">{node.name}</h4>
                        <span className="text-[10px] font-mono text-[#46707E]">
                          Bottleneck ID: {node.id}
                        </span>
                      </div>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-extrabold ${
                          isAiNode
                            ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                            : 'bg-rose-100 text-rose-800'
                        }`}
                      >
                        {isAiNode ? 'AI BOTTLENECK' : 'CHOKEPOINT'}
                      </span>
                    </div>

                    <div className="space-y-1.5 text-xs text-[#22393C]/80">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Chokepoint Delay:</span>
                        <span className="font-mono font-bold text-rose-600">{delay}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Intersection Velocity:</span>
                        <span className="font-mono font-bold text-[#22393C]">{speed}</span>
                      </div>
                      {isAiNode && latestTrafficTelemetry ? (
                        <>
                          <div className="flex justify-between">
                            <span className="text-gray-500">YOLO11n Vehicle Count:</span>
                            <span className="font-mono font-bold text-emerald-700">
                              {latestTrafficTelemetry.vehicleCount} ({latestTrafficTelemetry.trafficDensity})
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Vehicle Mix:</span>
                            <span className="font-mono font-medium text-[11px] text-[#22393C]">
                              {Object.entries(latestTrafficTelemetry.vehicleMix)
                                .map(([cls, pct]) => `${cls}: ${pct}`)
                                .join(', ')}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Telemetry Status:</span>
                            <span className="font-medium text-emerald-700 font-bold">
                              Live Edge AI Inferred
                            </span>
                          </div>
                        </>
                      ) : (
                        <div className="flex justify-between">
                          <span className="text-gray-500">Telemetry Status:</span>
                          <span className="font-medium text-emerald-700">PMPML Bus Synced</span>
                        </div>
                      )}
                    </div>
                  </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      </div>

      {/* Traffic Severity Legend */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-3 bg-white/70 backdrop-blur-md rounded-xl border border-[#46707E]/20 text-xs font-semibold text-[#22393C]">
        <div className="flex items-center space-x-2">
          <span className="text-gray-500 font-bold uppercase tracking-wider text-[11px]">
            Congestion Intensity Legend:
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-5">
          <div className="flex items-center space-x-2">
            <span className="w-5 h-2 rounded-full bg-[#16A34A] inline-block shadow-sm"></span>
            <span>Free Flow (&gt;35 km/h, &lt;3 min delay)</span>
          </div>

          <div className="flex items-center space-x-2">
            <span className="w-5 h-2 rounded-full bg-[#F59E0B] inline-block shadow-sm"></span>
            <span>Moderate (18–35 km/h, 3–10 min delay)</span>
          </div>

          <div className="flex items-center space-x-2">
            <span className="w-5 h-2 rounded-full bg-[#DC2626] inline-block shadow-sm"></span>
            <span>High / Severe (&lt;18 km/h, &gt;10 min delay)</span>
          </div>

          <div className="flex items-center space-x-2">
            <span className="w-3.5 h-3.5 rounded-full border-2 border-rose-600 bg-[#22393C] flex items-center justify-center text-[9px] text-white">
              ⚠️
            </span>
            <span>Active Chokepoint / Bottleneck</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export type NavigationModule =
  | 'edge-ai'
  | 'event-engine'
  | 'dashboard'
  | 'fleet'
  | 'event-log'
  | 'infrastructure'
  | 'traffic'
  | 'safety';

export type EventCategory =
  | 'Pothole'
  | 'Road Damage'
  | 'Traffic'
  | 'Vehicle'
  | 'Safety'
  | 'Safety / Pedestrian Risk'
  | 'Accidents / Incidents'
  | 'Waterlogging'
  | 'Infrastructure Deficiencies'
  | 'ANPR / OCR Events';

export type IssueStatus = 'New' | 'Confirmed' | 'In Progress' | 'Resolved';
export type SeverityLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface Bus {
  id: string;             // e.g. "BUS-01"
  route: string;          // e.g. "Route 17"
  status: 'Online' | 'Offline';
  driverId: string;       // e.g. "D-1042"
  speed: string;          // e.g. "40 km/hr" or "-"
  lastUpdated: string;    // e.g. "10:42 AM"
  location: string;       // e.g. "MG Road"
  openIssues: number;     // e.g. 3
  coordinates: [number, number]; // [lat, lng]
  distanceToday: string;  // e.g. "142 km"
  totalEvents: number;    // e.g. 28
  activeCamera: string;
}

export type ObservationType = 'CONFIRMED' | 'NON_CONFIRMING';
export type CorroborationStatus =
  | 'SINGLE_BUS'
  | 'CORROBORATED'
  | 'VERIFIED'
  | 'NEEDS_VERIFICATION'
  | 'POTENTIAL_FALSE_POSITIVE';

export interface BusObservation {
  id: string;
  busId: string;
  route?: string;
  timestamp: string;
  observationType: ObservationType;
  confidence?: number;
  distanceMeters?: number;
  evidenceUrl?: string;
  notes?: string;
}

export interface UrbanEvent {
  id: string;             // e.g. "EVT-000184"
  type: string;           // e.g. "ROAD_DAMAGE"
  category: EventCategory | string;
  className: string;      // e.g. "Pothole"
  confidence: number;     // 0 - 100 or 0.0 - 1.0
  severity: SeverityLevel;
  busId: string;          // e.g. "BUS-01"
  timestamp: string;      // e.g. "2026-04-22 10:32:14"
  timeFormatted: string;  // e.g. "10:32 AM"
  locationName: string;   // e.g. "MG Road, Pune"
  coordinates: [number, number];
  evidenceFrame: string;  // e.g. "frame_1024.jpg"
  evidenceUrl?: string;   // e.g. "/api/evidence/evidence_EVT-000717.jpg"
  evidenceDataUri?: string; // base64 fallback data URI
  status: IssueStatus;
  contextNote?: string;
  isTransmitted?: boolean;
  sourceFilename?: string;
  frameNumber?: number;
  vehicleId?: string;
  uniqueCount?: number;
  uniqueCounts?: Record<string, number>;
  vehicleCount?: number;
  trafficDensity?: string;
  congestionLevel?: string;
  vehicleMix?: Record<string, string>;
  pedestrianCount?: number;
  riskScore?: number;
  riskLevel?: string;
  riskEstimateType?: string;
  maxProximityRatio?: number;
  minCenterOffset?: number;
  inConflictZoneCount?: number;
  numberPlate?: string;
  ocrConfidence?: number;
  plateDetected?: boolean;
  plateStatus?: string;
  isReliablePlate?: boolean;
  candidateSource?: string;
  vehicleClass?: string;
  rashStatus?: string;
  hitAndRunStatus?: string;
  suspectVehicleId?: string;
  impactedVehicleId?: string;
  involvedVehicles?: Array<{ trackId: number; vehicleClass: string; role: 'suspect' | 'impacted'; plateNumber?: string; ocrConfidence?: number }>;
  triggeredRules?: string[];
  motionMetrics?: Record<string, any>;
  interactionMetrics?: Record<string, any>;
  isLiveDetection?: boolean;
  corroborationStatus?: CorroborationStatus;
  corroborationCount?: number;
  nonConfirmationCount?: number;
  participatingBuses?: string[];
  observations?: BusObservation[];
}


export interface TrafficItem {
  id: string;
  location: string;       // e.g. "University Road"
  type: 'Bottleneck' | 'Route Delay';
  currentStatus: string;  // e.g. "Heavy Congestion"
  severity: 'High' | 'Medium' | 'Low';
  avgDelay: string;       // e.g. "14 min"
  persistence: string;    // e.g. "28 min"
  coordinates: [number, number];
  speedKmh: number;
}

export interface SafetyItem {
  id: string;
  vehicleId: string;      // e.g. "V-108"
  numberPlate: string;    // e.g. "MH12AB1234"
  ocrConfidence: number;  // e.g. 94
  plateStatus?: string;
  isReliablePlate?: boolean;
  candidateSource?: string;
  location: string;       // e.g. "University Road"
  timestamp: string;      // e.g. "10:41:23"
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  type: 'Rash Driving' | 'Hit & Run' | 'Pedestrian Risk' | 'ANPR / OCR';
  contextDetails: string[]; // e.g. ["high speed (78 km/h)", "abrupt lane change", "dangerous proximity"]
  evidenceUrl?: string;
  evidenceDataUri?: string;
  pedestrianCount?: number;
  riskScore?: number;
  riskLevel?: string;
  riskEstimateType?: string;
  maxProximityRatio?: number;
  minCenterOffset?: number;
  inConflictZoneCount?: number;
  triggeredRules?: string[];
  motionMetrics?: Record<string, any>;
  interactionMetrics?: Record<string, any>;
  hitAndRunStatus?: string;
  suspectVehicleId?: string;
  impactedVehicleId?: string;
  involvedVehicles?: Array<{ trackId: number; vehicleClass: string; role: 'suspect' | 'impacted'; plateNumber?: string; ocrConfidence?: number }>;
  coordinates?: [number, number];
}


export interface NotificationItem {
  id: string;
  title: string;
  message: string;
  timestamp: string;
  category?: string;
}

export interface MapLayerFilterState {
  buses: boolean;
  roadDamage: boolean;
  traffic: boolean;
  vehicles: boolean;
  safety: boolean;
  accidents: boolean;
  waterlogging: boolean;
  infrastructure: boolean;
  anprOcr: boolean;
}

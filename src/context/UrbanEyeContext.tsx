import React, { createContext, useContext, useState } from 'react';
import {
  NavigationModule,
  Bus,
  UrbanEvent,
  TrafficItem,
  SafetyItem,
  NotificationItem,
  IssueStatus,
  CorroborationStatus,
  BusObservation,
} from '../types';
import {
  initialBuses,
  initialEvents,
  trafficItems,
  safetyItems,
  initialEventCategoryStats,
} from '../data/mockData';

// Production API base configuration for deployed environments
const API_BASE = ((import.meta.env.VITE_API_URL as string) || '').replace(/\/$/, '');

export const getApiUrl = (endpoint: string): string => {
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return API_BASE ? `${API_BASE}${cleanEndpoint}` : cleanEndpoint;
};

export const resolveEvidenceUrl = (url?: string): string | undefined => {
  if (!url) return undefined;
  if (url.startsWith('http://') || url.startsWith('https://') || url.startsWith('data:')) {
    return url;
  }
  const clean = url.startsWith('/') ? url : `/${url}`;
  return API_BASE ? `${API_BASE}${clean}` : clean;
};

interface UrbanEyeContextType {
  currentModule: NavigationModule;
  setCurrentModule: (module: NavigationModule) => void;
  buses: Bus[];
  selectedBusId: string;
  setSelectedBusId: (busId: string) => void;
  events: UrbanEvent[];
  selectedEventId: string | null;
  setSelectedEventId: (id: string | null) => void;
  traffic: TrafficItem[];
  safety: SafetyItem[];
  setSafety: React.Dispatch<React.SetStateAction<SafetyItem[]>>;
  categoryStats: typeof initialEventCategoryStats;
  notifications: NotificationItem[];
  dismissNotification: (id: string) => void;
  generatedEngineEvent: UrbanEvent;
  pipelineState: {
    isRunning: boolean;
    currentStep: number;
    stepLabels: string[];
    detections: { label: string; conf: number; type: string }[];
  };
  triggerEdgeProcessing: (busId?: string, videoFile?: File | null, cameraName?: string) => Promise<void>;
  updateEventStatus: (eventId: string, newStatus: IssueStatus) => void;
  corroborateEvent: (eventId: string, busId: string, confirmed: boolean, notes?: string) => void;
  navigateToEventLogWithEvent: (eventId: string) => void;
  uploadedFile: File | null;
  setUploadedFile: (file: File | null) => void;
  uploadedFileName: string | null;
  setUploadedFileName: (name: string | null) => void;
  uniqueVehicleCounts: { Car: number; Motorcycle: number; Bus: number; Truck: number; total: number } | null;
  latestTrafficTelemetry: {
    vehicleCount: number;
    trafficDensity: string;
    congestionLevel: string;
    vehicleMix: Record<string, string>;
    occupancyRatio: number;
    timestamp: string;
  } | null;
  latestPedestrianRiskTelemetry: {
    pedestrianCount: number;
    riskScore: number;
    riskLevel: string;
    riskEstimateType: string;
    maxProximityRatio: number;
    minCenterOffset: number;
    inConflictZoneCount: number;
    timestamp: string;
  } | null;
  latestRashDrivingTelemetry: {
    vehicleId: string;
    vehicleClass: string;
    riskScore: number;
    status: string;
    severity: string;
    triggeredRules: string[];
    motionMetrics: Record<string, any>;
    timestamp: string;
  } | null;
  latestHitAndRunTelemetry: {
    suspectVehicleId: string;
    impactedVehicleId: string;
    vehicleClass: string;
    riskScore: number;
    status: string;
    severity: string;
    triggeredRules: string[];
    interactionMetrics: Record<string, any>;
    timestamp: string;
  } | null;
  setTraffic: React.Dispatch<React.SetStateAction<TrafficItem[]>>;
}


const UrbanEyeContext = createContext<UrbanEyeContextType | undefined>(undefined);

const PIPELINE_STEPS = [
  'Video Input',
  'Frame Processing',
  'Object Detection (YOLO)',
  'Tracking (ByteTrack)',
  'Event Engine',
  'Event Transmission',
];

export const UrbanEyeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentModule, setCurrentModule] = useState<NavigationModule>('dashboard');
  const [buses, setBuses] = useState<Bus[]>(initialBuses);
  const [selectedBusId, setSelectedBusId] = useState<string>('BUS-01');
  const [events, setEvents] = useState<UrbanEvent[]>(initialEvents);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [traffic, setTraffic] = useState<TrafficItem[]>(trafficItems);
  const [safety, setSafety] = useState<SafetyItem[]>(safetyItems);
  const [categoryStats, setCategoryStats] = useState(initialEventCategoryStats);
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [generatedEngineEvent, setGeneratedEngineEvent] = useState<UrbanEvent>(initialEvents[0]);
  const [uniqueVehicleCounts, setUniqueVehicleCounts] = useState<{
    Car: number;
    Motorcycle: number;
    Bus: number;
    Truck: number;
    total: number;
  } | null>(null);
  const [latestTrafficTelemetry, setLatestTrafficTelemetry] = useState<{
    vehicleCount: number;
    trafficDensity: string;
    congestionLevel: string;
    vehicleMix: Record<string, string>;
    occupancyRatio: number;
    timestamp: string;
  } | null>(null);
  const [latestPedestrianRiskTelemetry, setLatestPedestrianRiskTelemetry] = useState<{
    pedestrianCount: number;
    riskScore: number;
    riskLevel: string;
    riskEstimateType: string;
    maxProximityRatio: number;
    minCenterOffset: number;
    inConflictZoneCount: number;
    timestamp: string;
  } | null>(null);
  const [latestRashDrivingTelemetry, setLatestRashDrivingTelemetry] = useState<{
    vehicleId: string;
    vehicleClass: string;
    riskScore: number;
    status: string;
    severity: string;
    triggeredRules: string[];
    motionMetrics: Record<string, any>;
    timestamp: string;
  } | null>(null);
  const [latestHitAndRunTelemetry, setLatestHitAndRunTelemetry] = useState<{
    suspectVehicleId: string;
    impactedVehicleId: string;
    vehicleClass: string;
    riskScore: number;
    status: string;
    severity: string;
    triggeredRules: string[];
    interactionMetrics: Record<string, any>;
    timestamp: string;
  } | null>(null);

  const [pipelineState, setPipelineState] = useState({
    isRunning: false,
    currentStep: 6, // all green by default
    stepLabels: PIPELINE_STEPS,
    detections: [
      { label: 'Car', conf: 0.91, type: 'vehicle' },
      { label: 'Two-wheeler', conf: 0.87, type: 'vehicle' },
      { label: 'Pothole', conf: 0.94, type: 'damage' },
      { label: 'Truck', conf: 0.76, type: 'vehicle' },
    ],
  });

  const dismissNotification = (id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const updateEventStatus = (eventId: string, newStatus: IssueStatus) => {
    setEvents((prev) =>
      prev.map((evt) => (evt.id === eventId ? { ...evt, status: newStatus } : evt))
    );
  };

  const corroborateEvent = (
    eventId: string,
    busId: string,
    confirmed: boolean,
    notes?: string
  ) => {
    setEvents((prevEvents) => {
      const target = prevEvents.find((e) => e.id === eventId);
      if (!target) return prevEvents;

      const currentConf = target.confidence || 85;
      const newConf = confirmed
        ? Math.min(99, currentConf + 6)
        : Math.max(20, currentConf - 12);

      const busObj = buses.find((b) => b.id === busId);
      const busRoute = busObj?.route || 'Route 17';
      const now = new Date();
      const timeFormatted = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const timestampStr = now.toISOString().replace('T', ' ').substring(0, 19);

      const initialObs: BusObservation = {
        id: `OBS-${target.id}-01`,
        busId: target.busId,
        route: buses.find((b) => b.id === target.busId)?.route || 'Route 17',
        timestamp: target.timestamp,
        observationType: 'CONFIRMED',
        confidence: currentConf,
        distanceMeters: 0,
        notes: `Initial edge AI pass confirmation by ${target.busId}`,
      };

      const currentObs: BusObservation[] = target.observations && target.observations.length > 0
        ? target.observations
        : [initialObs];

      const newObs: BusObservation = {
        id: `OBS-${target.id}-${(currentObs.length + 1).toString().padStart(2, '0')}`,
        busId,
        route: busRoute,
        timestamp: timestampStr,
        observationType: confirmed ? 'CONFIRMED' : 'NON_CONFIRMING',
        confidence: newConf,
        distanceMeters: 0,
        notes:
          notes ||
          (confirmed
            ? `Corroborating pass by ${busId} (${busRoute}): defect confirmed. Confidence +6%`
            : `Non-confirming pass by ${busId} (${busRoute}): defect not visible. Confidence -12%`),
      };

      const updatedObs = [...currentObs, newObs];
      const existingBuses = target.participatingBuses || [target.busId];
      const updatedBuses = confirmed && !existingBuses.includes(busId)
        ? [...existingBuses, busId]
        : existingBuses;

      const currentNonConf = target.nonConfirmationCount || 0;
      const newNonConf = confirmed ? currentNonConf : currentNonConf + 1;
      const newCorrobCount = updatedBuses.length;

      let newStatus: CorroborationStatus = target.corroborationStatus || 'SINGLE_BUS';
      if (!confirmed) {
        if (newNonConf >= 2) {
          newStatus = 'POTENTIAL_FALSE_POSITIVE';
        } else {
          newStatus = 'NEEDS_VERIFICATION';
        }
      } else {
        if (newCorrobCount >= 3) {
          newStatus = 'VERIFIED';
        } else if (newCorrobCount >= 2) {
          newStatus = 'CORROBORATED';
        } else {
          newStatus = 'SINGLE_BUS';
        }
      }

      const updatedEvt: UrbanEvent = {
        ...target,
        confidence: newConf,
        corroborationStatus: newStatus,
        corroborationCount: newCorrobCount,
        nonConfirmationCount: newNonConf,
        participatingBuses: updatedBuses,
        observations: updatedObs,
      };

      if (generatedEngineEvent.id === target.id) {
        setGeneratedEngineEvent(updatedEvt);
      }

      // Add feedback notification
      const toastTitle = confirmed
        ? `Multi-Bus Corroboration: ${target.className}`
        : `Non-Confirmation Alert: ${target.className}`;
      const toastMsg = confirmed
        ? `Bus ${busId} corroborated defect at ${target.locationName}. Status: ${newStatus} (${newConf}% conf).`
        : `Bus ${busId} did not confirm defect at ${target.locationName}. Confidence reduced to ${newConf}%.`;

      const newToast: NotificationItem = {
        id: `toast-corrob-${Date.now()}-${Math.floor(Math.random() * 100000)}`,
        title: toastTitle,
        message: toastMsg,
        timestamp: 'Just now',
        category: target.category as string,
      };
      setNotifications((nPrev) => [newToast, ...nPrev]);
      setTimeout(() => dismissNotification(newToast.id), 6000);

      return prevEvents.map((e) => (e.id === eventId ? updatedEvt : e));
    });
  };

  const navigateToEventLogWithEvent = (eventId: string) => {
    setSelectedEventId(eventId);
    setCurrentModule('event-log');
  };

  const triggerEdgeProcessing = async (
    busId = 'BUS-01',
    videoFile?: File | null,
    cameraName = 'Front Camera'
  ) => {
    if (pipelineState.isRunning) return;

    const fileToProcess = videoFile !== undefined ? videoFile : uploadedFile;

    // Start pipeline
    setPipelineState((prev) => ({ ...prev, isRunning: true, currentStep: 1 }));

    // Step 2: Frame Processing
    await new Promise((r) => setTimeout(r, 600));
    setPipelineState((prev) => ({ ...prev, currentStep: 2 }));

    // Step 3: Object Detection (Roboflow Inference)
    await new Promise((r) => setTimeout(r, 600));
    setPipelineState((prev) => ({ ...prev, currentStep: 3 }));

    let realEvent: UrbanEvent | null = null;
    let customDetections: { label: string; conf: number; type: string }[] = [];

    try {
      if (fileToProcess) {
        // Real video/image uploaded by user
        const formData = new FormData();
        formData.append('file', fileToProcess);
        formData.append('bus_id', busId);
        formData.append('camera_name', cameraName);
        formData.append('confidence_threshold', '0.25');

        const response = await fetch(getApiUrl('/api/upload'), {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.event) {
            realEvent = {
              ...(data.event as UrbanEvent),
              evidenceUrl: resolveEvidenceUrl(data.event.evidenceUrl || data.evidence_url),
              evidenceDataUri: data.event.evidenceDataUri || data.evidence_image,
            };
            if (data.detections && data.detections.length > 0) {
              customDetections = data.detections;
            }
            if (data.unique_counts) {
              setUniqueVehicleCounts(data.unique_counts);
            }
            if (data.traffic_detected) {
              const tel = {
                vehicleCount: data.vehicle_count || (realEvent ? realEvent.vehicleCount : 12) || 12,
                trafficDensity: data.traffic_density || (realEvent ? realEvent.trafficDensity : 'HIGH') || 'HIGH',
                congestionLevel: data.congestion_level || (realEvent ? realEvent.congestionLevel : 'HIGH') || 'HIGH',
                vehicleMix: data.vehicle_mix || (realEvent ? realEvent.vehicleMix : {}) || {},
                occupancyRatio: data.occupancy_ratio || 0.0,
                timestamp: realEvent ? realEvent.timeFormatted : 'Just now',
              };
              setLatestTrafficTelemetry(tel);

              const newTrafficItem: TrafficItem = {
                id: `TRF-${realEvent ? realEvent.id.replace('EVT-', '') : Date.now()}`,
                location: realEvent?.locationName || 'Ahmednagar Road / Viman Nagar, Pune',
                type: 'Bottleneck',
                currentStatus: `${tel.congestionLevel} Congestion (${tel.trafficDensity} Density)`,
                severity: (tel.congestionLevel === 'SEVERE' || tel.congestionLevel === 'HIGH') ? 'High' : (tel.congestionLevel === 'MODERATE' ? 'Medium' : 'Low'),
                avgDelay: tel.congestionLevel === 'SEVERE' ? '24 min' : tel.congestionLevel === 'HIGH' ? '16 min' : '8 min',
                persistence: '18 min',
                coordinates: realEvent?.coordinates || [18.5679, 73.9143],
                speedKmh: tel.congestionLevel === 'SEVERE' ? 12 : tel.congestionLevel === 'HIGH' ? 18 : 35,
              };
              setTraffic((prev) => [newTrafficItem, ...prev.filter((t) => t.id !== newTrafficItem.id)]);
            }
            if (data.pedestrian_risk_detected) {
              const pedTel = {
                pedestrianCount: data.pedestrian_count || realEvent?.pedestrianCount || 0,
                riskScore: data.risk_score || realEvent?.riskScore || 0,
                riskLevel: data.risk_level || realEvent?.riskLevel || 'HIGH',
                riskEstimateType: data.risk_estimate_type || 'Prototype Image-Based Risk Estimate',
                maxProximityRatio: data.max_proximity_ratio || realEvent?.maxProximityRatio || 0.0,
                minCenterOffset: data.min_center_offset || realEvent?.minCenterOffset || 0.0,
                inConflictZoneCount: data.in_conflict_zone_count || realEvent?.inConflictZoneCount || 0,
                timestamp: realEvent ? realEvent.timeFormatted : 'Just now',
              };
              setLatestPedestrianRiskTelemetry(pedTel);

              const newSafetyItem: SafetyItem = {
                id: `SAF-${realEvent ? realEvent.id.replace('EVT-', '') : Date.now()}`,
                vehicleId: busId,
                numberPlate: `BUS-${busId.replace('BUS-', '')}`,
                ocrConfidence: realEvent?.confidence || 88,
                location: realEvent?.locationName || 'MG Road, Pune',
                timestamp: realEvent?.timeFormatted || 'Just now',
                severity: (realEvent?.severity || 'HIGH') as 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
                type: 'Pedestrian Risk',
                contextDetails: [
                  `Prototype Image-Based Risk Estimate: ${pedTel.riskScore}/100 (${pedTel.riskLevel})`,
                  `${pedTel.pedestrianCount} Pedestrian(s) identified in front roadway view`,
                  `${pedTel.inConflictZoneCount} Pedestrian(s) in primary travel corridor`,
                  `Camera Bumper Proximity: ${Math.round((pedTel.maxProximityRatio || 0.7) * 100)}%`,
                ],
                evidenceUrl: realEvent?.evidenceUrl || data.evidence_url,
                evidenceDataUri: realEvent?.evidenceDataUri || data.evidence_image,
                pedestrianCount: pedTel.pedestrianCount,
                riskScore: pedTel.riskScore,
                riskLevel: pedTel.riskLevel,
                riskEstimateType: pedTel.riskEstimateType,
                maxProximityRatio: pedTel.maxProximityRatio,
                minCenterOffset: pedTel.minCenterOffset,
                inConflictZoneCount: pedTel.inConflictZoneCount,
                coordinates: realEvent?.coordinates,
              };
              setSafety((prev) => [newSafetyItem, ...prev.filter((s) => s.id !== newSafetyItem.id)]);
            }
            if (data.anpr_detected || realEvent?.type === 'ANPR_OCR') {
              const plateText = data.plate_number || realEvent?.numberPlate || 'MH12AB1234';
              const ocrConf = data.ocr_confidence || realEvent?.confidence || 85;
              const vClass = data.vehicle_class || 'Vehicle';
              const vId = data.vehicle_id || realEvent?.vehicleId || `V-${plateText}`;
              const plateStatus = data.plate_status || realEvent?.plateStatus || 'RELIABLE PLATE';
              const isReliable = data.is_reliable !== undefined ? data.is_reliable : (realEvent?.isReliablePlate ?? true);
              const candidateSource = realEvent?.candidateSource || 'morphological_contour';

              const newAnprSafetyItem: SafetyItem = {
                id: `SAF-${realEvent ? realEvent.id.replace('EVT-', '') : Date.now()}`,
                vehicleId: vId,
                numberPlate: plateText,
                ocrConfidence: ocrConf,
                plateStatus,
                isReliablePlate: isReliable,
                candidateSource,
                location: realEvent?.locationName || 'MG Road, Pune',
                timestamp: realEvent?.timeFormatted || 'Just now',
                severity: (realEvent?.severity || 'HIGH') as 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
                type: 'ANPR / OCR',
                contextDetails: [
                  `Live Neural ANPR Recognition: ${vClass} plate "${plateText}" (${ocrConf}% OCR confidence, ${plateStatus})`,
                  `Multi-Stage Candidate Extraction via YOLO11n + Morphological Priors + EasyOCR CRAFT`,
                  `Real Edge AI evidence logged on Bus ${busId}`,
                ],
                evidenceUrl: realEvent?.evidenceUrl || data.evidence_url,
                evidenceDataUri: realEvent?.evidenceDataUri || data.evidence_image,
                coordinates: realEvent?.coordinates,
              };
              setSafety((prev) => [newAnprSafetyItem, ...prev.filter((s) => s.id !== newAnprSafetyItem.id)]);
            }
            if (data.rash_driving_detected || realEvent?.type === 'RASH_DRIVING') {
              const bestCand = data.best_candidate || {};
              const vClass = bestCand.vehicle_class || realEvent?.vehicleClass || 'Vehicle';
              const tid = bestCand.track_id !== undefined ? bestCand.track_id : (realEvent?.vehicleId?.replace('#', '') || '1');
              const riskScore = data.risk_score !== undefined ? data.risk_score : (realEvent?.riskScore || 75);
              const status = data.status || realEvent?.rashStatus || 'RASH DRIVING';
              const severity = (data.severity || realEvent?.severity || 'HIGH') as 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
              const rules: string[] = data.triggered_rules || realEvent?.triggeredRules || [];
              const motionMetrics = data.motion_metrics || realEvent?.motionMetrics || {};

              const rashTel = {
                vehicleId: `V-${tid}`,
                vehicleClass: vClass,
                riskScore: riskScore,
                status: status,
                severity: severity,
                triggeredRules: rules,
                motionMetrics: motionMetrics,
                timestamp: realEvent ? realEvent.timeFormatted : 'Just now',
              };
              setLatestRashDrivingTelemetry(rashTel);

              const newRashSafetyItem: SafetyItem = {
                id: `SAF-${realEvent ? realEvent.id.replace('EVT-', '') : Date.now()}`,
                vehicleId: `V-${tid}`,
                numberPlate: `MH12-TRK-${tid}`,
                ocrConfidence: Math.max(50, riskScore),
                location: realEvent?.locationName || 'FC Road, Pune',
                timestamp: realEvent?.timeFormatted || 'Just now',
                severity: severity,
                type: 'Rash Driving',
                contextDetails: [
                  `Deterministic Kinematic Risk Score: ${riskScore}/100 (${status})`,
                  rules.length > 0 ? `Triggered Rules: ${rules.join(', ')}` : 'Stable trajectory motion parameters',
                  `Relative Motion: Speed ${motionMetrics.max_speed_proxy || 0} px/f | Max Swerve: ${motionMetrics.max_swerve_deg || 0}° | Reversals: ${motionMetrics.zigzag_reversals || 0}`,
                  `ByteTrack ID #${tid} persistent trajectory tracking across frames`,
                ],
                evidenceUrl: realEvent?.evidenceUrl || data.evidence_url,
                evidenceDataUri: realEvent?.evidenceDataUri || data.evidence_image,
                riskScore: riskScore,
                riskLevel: severity,
                triggeredRules: rules,
                motionMetrics: motionMetrics,
                coordinates: realEvent?.coordinates,
              };
              setSafety((prev) => [newRashSafetyItem, ...prev.filter((s) => s.id !== newRashSafetyItem.id)]);
            }
            if (data.hit_and_run_detected || realEvent?.type === 'HIT_AND_RUN') {
              const bestCand = data.best_candidate || {};
              const vClassA = bestCand.vehicle_class_a || realEvent?.vehicleClass || 'Vehicle';
              const vClassB = bestCand.vehicle_class_b || 'Vehicle';
              const tidA = bestCand.track_id_a !== undefined ? bestCand.track_id_a : (realEvent?.suspectVehicleId?.replace('#', '') || '1');
              const tidB = bestCand.track_id_b !== undefined ? bestCand.track_id_b : (realEvent?.impactedVehicleId?.replace('#', '') || '2');
              const riskScore = data.risk_score !== undefined ? data.risk_score : (realEvent?.riskScore || 75);
              const status = data.status || realEvent?.hitAndRunStatus || 'HIT & RUN CANDIDATE';
              const severity = (data.severity || realEvent?.severity || 'CRITICAL') as 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
              const rules: string[] = data.triggered_rules || realEvent?.triggeredRules || [];
              const interactionMetrics = data.interaction_metrics || realEvent?.interactionMetrics || {};
              const plateText = realEvent?.numberPlate || `MH12-TRK-${tidA}`;
              const ocrConf = realEvent?.ocrConfidence || Math.max(50, riskScore);

              const hitRunTel = {
                suspectVehicleId: `V-${tidA}`,
                impactedVehicleId: `V-${tidB}`,
                vehicleClass: vClassA,
                riskScore: riskScore,
                status: status,
                severity: severity,
                triggeredRules: rules,
                interactionMetrics: interactionMetrics,
                timestamp: realEvent ? realEvent.timeFormatted : 'Just now',
              };
              setLatestHitAndRunTelemetry(hitRunTel);

              const newHitRunSafetyItem: SafetyItem = {
                id: `SAF-${realEvent ? realEvent.id.replace('EVT-', '') : Date.now()}`,
                vehicleId: `V-${tidA}`,
                numberPlate: plateText,
                ocrConfidence: ocrConf,
                location: realEvent?.locationName || 'Swargate Junction, Pune',
                timestamp: realEvent?.timeFormatted || 'Just now',
                severity: severity,
                type: 'Hit & Run',
                suspectVehicleId: `V-${tidA}`,
                impactedVehicleId: `V-${tidB}`,
                hitAndRunStatus: status,
                triggeredRules: rules,
                interactionMetrics: interactionMetrics,
                contextDetails: [
                  `Multi-Signal Collision Heuristic: Risk Score ${riskScore}/100 (${status})`,
                  `Suspect Vehicle: ${vClassA} #${tidA} vs Impacted: ${vClassB} #${tidB}`,
                  rules.length > 0 ? `Triggered Rules: ${rules.join(', ')}` : 'Spatial proximity & trajectory divergence',
                  `Proximity: ${interactionMetrics.min_distance_px || 0} px | Fleeing Speed: ${interactionMetrics.post_interaction_speed_a || 0} px/f`,
                ],
                evidenceUrl: realEvent?.evidenceUrl || data.evidence_url,
                evidenceDataUri: realEvent?.evidenceDataUri || data.evidence_image,
                riskScore: riskScore,
                riskLevel: severity,
                coordinates: realEvent?.coordinates,
              };
              setSafety((prev) => [newHitRunSafetyItem, ...prev.filter((s) => s.id !== newHitRunSafetyItem.id)]);
            }
            // Immediately update generatedEngineEvent and events so Event Engine reflects it
            setGeneratedEngineEvent(realEvent);
            setEvents((prev) => [realEvent!, ...prev.filter((e) => e.id !== realEvent!.id)]);
          }
        }

      } else {
        // Trigger edge demo inference with Roboflow
        const formData = new FormData();
        formData.append('bus_id', busId);
        formData.append('camera_name', cameraName);

        const response = await fetch(getApiUrl('/api/demo-inference'), {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.potholes_detected && data.event) {
            realEvent = {
              ...(data.event as UrbanEvent),
              evidenceUrl: resolveEvidenceUrl(data.event.evidenceUrl || data.evidence_url),
              evidenceDataUri: data.event.evidenceDataUri || data.evidence_image,
            };
            if (data.detections && data.detections.length > 0) {
              customDetections = data.detections;
            }
            // Immediately update generatedEngineEvent and events
            setGeneratedEngineEvent(realEvent);
            setEvents((prev) => [realEvent!, ...prev.filter((e) => e.id !== realEvent!.id)]);
          }
        }
      }
    } catch (err) {
      console.warn('Backend inference fallback active:', err);
    }

    // Step 4: Tracking (ByteTrack)
    await new Promise((r) => setTimeout(r, 600));
    setPipelineState((prev) => ({
      ...prev,
      currentStep: 4,
      detections: customDetections,
    }));

    // Step 5: Event Engine (Schema Normalization)
    await new Promise((r) => setTimeout(r, 600));
    setPipelineState((prev) => ({ ...prev, currentStep: 5 }));

    // Step 6: Event Transmission & Registration
    await new Promise((r) => setTimeout(r, 600));

    if (fileToProcess) {
      if (realEvent) {
        // Real event detected from uploaded video/image (damage or vehicle or traffic)
        const isNearby = (c1: [number, number], c2: [number, number]) => {
          const dLat = Math.abs(c1[0] - c2[0]);
          const dLng = Math.abs(c1[1] - c2[1]);
          const distMeters = Math.sqrt(Math.pow(dLat * 111000, 2) + Math.pow(dLng * 105000, 2));
          return distMeters <= 50;
        };

        const existingMatch = (realEvent.category === 'Pothole' || realEvent.category === 'Road Damage' || realEvent.category === 'Infrastructure Deficiencies')
          ? events.find(
              (e) =>
                e.category === realEvent!.category &&
                isNearby(e.coordinates, realEvent!.coordinates) &&
                e.busId !== busId
            )
          : null;

        if (existingMatch) {
          corroborateEvent(
            existingMatch.id,
            busId,
            true,
            `Edge AI detection on ${cameraName} by ${busId} corroborates ${existingMatch.id} (${existingMatch.className})`
          );
        } else {
          setEvents((prev) => [realEvent!, ...prev.filter((e) => e.id !== realEvent!.id)]);
          setGeneratedEngineEvent(realEvent);
        }

        const eventCat = realEvent.category || (
          realEvent.type === 'VEHICLE_DETECTION'
            ? 'Vehicle'
            : (realEvent.type === 'TRAFFIC_CONGESTION'
                ? 'Traffic'
                : (realEvent.type === 'PEDESTRIAN_RISK' ? 'Safety' : 'Pothole'))
        );

        setCategoryStats((prev) =>
          prev.map((stat) =>
            stat.category === eventCat ||
            (eventCat === 'Vehicle' && stat.category === 'Traffic') ||
            (eventCat === 'Safety' && (stat.category === 'Safety' || stat.category === 'Safety / Pedestrian Risk'))
              ? { ...stat, count: stat.count + 1, sparkline: [...stat.sparkline.slice(1), stat.sparkline[stat.sparkline.length - 1] + 2] }
              : stat
          )
        );

        setBuses((prev) =>
          prev.map((b) =>
            b.id === busId
              ? { ...b, openIssues: b.openIssues + 1, totalEvents: b.totalEvents + 1 }
              : b
          )
        );

        const newToast: NotificationItem = {
          id: `toast-${Date.now()}`,
          title:
            realEvent.type === 'TRAFFIC_CONGESTION'
              ? 'Traffic Alert Registered!'
              : realEvent.type === 'PEDESTRIAN_RISK'
              ? 'Pedestrian Risk Alert!'
              : realEvent.type === 'RASH_DRIVING'
              ? 'Rash Driving Incident Alert!'
              : realEvent.type === 'HIT_AND_RUN'
              ? 'Hit & Run Incident Alert!'
              : realEvent.type === 'ANPR_OCR'
              ? 'ANPR Plate Recognized!'
              : realEvent.type === 'INFRASTRUCTURE_DEFICIENCY'
              ? 'Infrastructure Deficiency Registered!'
              : 'Event Registered!',
          message: `${realEvent.className} detected (${busId}) registered via Edge AI (${realEvent.confidence}% conf).`,
          timestamp: 'Just now',
          category: eventCat,
        };
        setNotifications((prev) => [newToast, ...prev]);


        setTimeout(() => {
          dismissNotification(newToast.id);
        }, 7000);
      } else {
        // No defects or vehicles detected in uploaded media
        const newToast: NotificationItem = {
          id: `toast-${Date.now()}`,
          title: 'Analysis Complete',
          message: `Uploaded video processed (${busId}). No road surface defects or vehicles detected.`,
          timestamp: 'Just now',
          category: 'Road Damage',
        };
        setNotifications((prev) => [newToast, ...prev]);

        setTimeout(() => {
          dismissNotification(newToast.id);
        }, 7000);
      }
    } else {
      // Demo mode without user-uploaded file
      const newEvtId = `EVT-000${Math.floor(185 + Math.random() * 50)}`;
      const finalEvent: UrbanEvent = realEvent || {
        id: newEvtId,
        type: 'ROAD_DAMAGE',
        category: 'Pothole',
        className: 'Pothole',
        confidence: 94,
        severity: 'HIGH',
        busId: busId,
        timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19),
        timeFormatted: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        locationName: busId === 'BUS-02' ? 'FC Road, Pune' : 'MG Road, Pune',
        coordinates: [18.5204 + (Math.random() - 0.5) * 0.005, 73.8567 + (Math.random() - 0.5) * 0.005],
        evidenceFrame: 'frame_1024.jpg',
        status: 'New',
        contextNote: 'Fresh severe pothole detected by Edge AI front camera with 94% confidence.',
        isTransmitted: true,
      };

      setEvents((prev) => [finalEvent, ...prev.filter((e) => e.id !== finalEvent.id)]);
      setGeneratedEngineEvent(finalEvent);

      setCategoryStats((prev) =>
        prev.map((stat) =>
          stat.category === 'Pothole'
            ? { ...stat, count: stat.count + 1, sparkline: [...stat.sparkline.slice(1), stat.sparkline[stat.sparkline.length - 1] + 2] }
            : stat
        )
      );

      setBuses((prev) =>
        prev.map((b) =>
          b.id === busId
            ? { ...b, openIssues: b.openIssues + 1, totalEvents: b.totalEvents + 1 }
            : b
        )
      );

      const newToast: NotificationItem = {
        id: `toast-${Date.now()}`,
        title: 'Event Registered!',
        message: `Pothole detected (${busId}) registered via Roboflow AI (${finalEvent.confidence}% conf).`,
        timestamp: 'Just now',
        category: 'Pothole',
      };
      setNotifications((prev) => [newToast, ...prev]);

      setTimeout(() => {
        dismissNotification(newToast.id);
      }, 7000);
    }

    setPipelineState((prev) => ({ ...prev, isRunning: false, currentStep: 6 }));
  };

  return (
    <UrbanEyeContext.Provider
      value={{
        currentModule,
        setCurrentModule,
        buses,
        selectedBusId,
        setSelectedBusId,
        events,
        selectedEventId,
        setSelectedEventId,
        traffic,
        safety,
        setSafety,
        categoryStats,
        notifications,
        dismissNotification,
        generatedEngineEvent,
        pipelineState,
        triggerEdgeProcessing,
        updateEventStatus,
        corroborateEvent,
        navigateToEventLogWithEvent,
        uploadedFile,
        setUploadedFile,
        uploadedFileName,
        setUploadedFileName,
        uniqueVehicleCounts,
        latestTrafficTelemetry,
        latestPedestrianRiskTelemetry,
        latestRashDrivingTelemetry,
        latestHitAndRunTelemetry,
        setTraffic,
      }}

    >
      {children}
    </UrbanEyeContext.Provider>
  );
};

export const useUrbanEye = () => {
  const context = useContext(UrbanEyeContext);
  if (!context) {
    throw new Error('useUrbanEye must be used within an UrbanEyeProvider');
  }
  return context;
};

import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  Upload,
  Radio,
  CheckCircle2,
  AlertTriangle,
  Car,
  Truck,
  Bike,
  Bus,
  Maximize2,
  Volume2,
  Sparkles,
  Camera,
  User,
} from 'lucide-react';
import { useUrbanEye } from '../context/UrbanEyeContext';

export const EdgeAI: React.FC = () => {
  const {
    buses,
    selectedBusId,
    setSelectedBusId,
    pipelineState,
    triggerEdgeProcessing,
    uploadedFile,
    setUploadedFile,
    uploadedFileName,
    setUploadedFileName,
    uniqueVehicleCounts,
    latestTrafficTelemetry,
    latestPedestrianRiskTelemetry,
  } = useUrbanEye();

  const [activeCamera, setActiveCamera] = useState<'Front Camera' | 'Left Camera' | 'Right Camera' | 'Rear Camera'>('Front Camera');
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackTime, setPlaybackTime] = useState(868); // ~14m28s

  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<number>(60);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const currentBus = buses.find((b) => b.id === selectedBusId) || buses[0];

  // Set up real video or image preview when a media file is uploaded
  useEffect(() => {
    if (
      uploadedFile &&
      (uploadedFile.type.startsWith('video/') ||
        uploadedFile.name.toLowerCase().endsWith('.mp4') ||
        uploadedFile.name.toLowerCase().endsWith('.mov') ||
        uploadedFile.name.toLowerCase().endsWith('.webm') ||
        uploadedFile.name.toLowerCase().endsWith('.avi'))
    ) {
      const url = URL.createObjectURL(uploadedFile);
      setVideoSrc(url);
      setImageSrc(null);
      setPlaybackTime(0);
      setIsPlaying(true);
      return () => {
        URL.revokeObjectURL(url);
      };
    } else if (
      uploadedFile &&
      (uploadedFile.type.startsWith('image/') ||
        uploadedFile.name.toLowerCase().endsWith('.jpg') ||
        uploadedFile.name.toLowerCase().endsWith('.jpeg') ||
        uploadedFile.name.toLowerCase().endsWith('.png') ||
        uploadedFile.name.toLowerCase().endsWith('.webp'))
    ) {
      const url = URL.createObjectURL(uploadedFile);
      setImageSrc(url);
      setVideoSrc(null);
      setPlaybackTime(0);
      setIsPlaying(false);
      return () => {
        URL.revokeObjectURL(url);
      };
    } else if (!uploadedFile) {
      setVideoSrc(null);
      setImageSrc(null);
    }
  }, [uploadedFile]);

  // Format seconds to mm:ss
  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `00:${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s}`;
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setPlaybackTime(Math.floor(videoRef.current.currentTime));
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setVideoDuration(Math.floor(videoRef.current.duration) || 60);
    }
  };

  const handlePlayPause = () => {
    if (videoRef.current && videoSrc) {
      if (videoRef.current.paused) {
        videoRef.current.play().catch(() => {});
        setIsPlaying(true);
      } else {
        videoRef.current.pause();
        setIsPlaying(false);
      }
    } else {
      setIsPlaying(!isPlaying);
    }
  };

  const handleScrubberClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    if (videoRef.current && videoSrc) {
      const newTime = ratio * videoDuration;
      videoRef.current.currentTime = newTime;
      setPlaybackTime(Math.floor(newTime));
    } else {
      setPlaybackTime(Math.floor(ratio * 3600));
    }
  };

  // Canvas road animation with AI bounding boxes
  useEffect(() => {
    let animationFrameId: number;
    let offset = 0;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;

      // 1. Sky & Horizon
      const skyGrad = ctx.createLinearGradient(0, 0, 0, h * 0.45);
      skyGrad.addColorStop(0, '#54727C');
      skyGrad.addColorStop(1, '#8BA39E');
      ctx.fillStyle = skyGrad;
      ctx.fillRect(0, 0, w, h * 0.45);

      // Distant Pune skyline & trees
      ctx.fillStyle = '#405B60';
      ctx.beginPath();
      ctx.moveTo(0, h * 0.45);
      ctx.lineTo(w * 0.15, h * 0.38);
      ctx.lineTo(w * 0.28, h * 0.42);
      ctx.lineTo(w * 0.45, h * 0.36);
      ctx.lineTo(w * 0.65, h * 0.43);
      ctx.lineTo(w * 0.85, h * 0.37);
      ctx.lineTo(w, h * 0.45);
      ctx.fill();

      // 2. Road surface (asphalt)
      const roadGrad = ctx.createLinearGradient(0, h * 0.45, 0, h);
      roadGrad.addColorStop(0, '#38484B');
      roadGrad.addColorStop(1, '#1A2729');
      ctx.fillStyle = roadGrad;
      ctx.fillRect(0, h * 0.45, w, h * 0.55);

      // Road perspective lanes
      ctx.strokeStyle = '#AFBB98';
      ctx.lineWidth = 3;
      ctx.setLineDash([20, 20]);
      ctx.lineDashOffset = -offset;

      // Center divider
      ctx.beginPath();
      ctx.moveTo(w * 0.5, h * 0.45);
      ctx.lineTo(w * 0.5, h);
      ctx.stroke();

      // Left lane divider
      ctx.beginPath();
      ctx.moveTo(w * 0.5, h * 0.45);
      ctx.lineTo(w * 0.18, h);
      ctx.stroke();

      // Right lane divider
      ctx.beginPath();
      ctx.moveTo(w * 0.5, h * 0.45);
      ctx.lineTo(w * 0.82, h);
      ctx.stroke();
      ctx.setLineDash([]); // reset

      // Road edges
      ctx.strokeStyle = '#CE CDB9';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(w * 0.5, h * 0.45);
      ctx.lineTo(0, h);
      ctx.moveTo(w * 0.5, h * 0.45);
      ctx.lineTo(w, h);
      ctx.stroke();

      // 3. AI Bounding Box Detections overlay
      // Detection 1: Ahead Car
      drawBBox(ctx, w * 0.54, h * 0.52, 90, 65, 'Car 0.91', '#46707E');

      // Detection 2: Two-wheeler (Scooter/Bike) on left
      drawBBox(ctx, w * 0.28, h * 0.58, 48, 70, 'Two-wheeler 0.87', '#6B8B81');

      // Detection 3: Pothole / Road hazard right ahead
      drawBBox(ctx, w * 0.42, h * 0.74, 110, 52, 'Pothole 0.94', '#E11D48', true);

      // Detection 4: Truck in distance
      drawBBox(ctx, w * 0.68, h * 0.47, 85, 80, 'Truck 0.76', '#AFBB98');

      // Advance road motion if playing
      if (isPlaying) {
        offset = (offset + 3.5) % 40;
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isPlaying]);

  // Helper to draw YOLO style bounding box
  const drawBBox = (
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    label: string,
    color: string,
    pulse = false
  ) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x, y, w, h);

    // Corner crosshairs
    const cLen = 8;
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    // Top-left
    ctx.moveTo(x, y + cLen);
    ctx.lineTo(x, y);
    ctx.lineTo(x + cLen, y);
    // Top-right
    ctx.moveTo(x + w - cLen, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + cLen);
    // Bottom-left
    ctx.moveTo(x, y + h - cLen);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + cLen, y + h);
    // Bottom-right
    ctx.moveTo(x + w - cLen, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - cLen);
    ctx.stroke();

    // Label tag
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 20, ctx.measureText(label).width + 16, 20);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px JetBrains Mono, sans-serif';
    ctx.fillText(label, x + 6, y - 6);

    // Hazard glow indicator
    if (pulse) {
      ctx.fillStyle = 'rgba(225, 29, 72, 0.15)';
      ctx.fillRect(x, y, w, h);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setUploadedFileName(file.name);
    }
    e.target.value = '';
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* Top Bus & Telemetry Header Bar */}
      <div className="glass-panel p-5 rounded-2xl flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-6">
          {/* Bus ID Dropdown */}
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold uppercase tracking-wider text-[#46707E]">
              Bus ID:
            </span>
            <select
              value={selectedBusId}
              onChange={(e) => setSelectedBusId(e.target.value)}
              className="bg-white/90 border border-[#46707E]/30 text-[#22393C] font-bold text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-[#46707E]"
            >
              {buses.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.id} ({b.route})
                </option>
              ))}
            </select>
          </div>

          {/* Camera Status */}
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-500 font-medium">Camera Status:</span>
            <span className="flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>ONLINE (1080p 30fps)</span>
            </span>
          </div>

          {/* GPS Coordinates */}
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-500 font-medium">GPS:</span>
            <span className="font-mono text-xs font-semibold text-[#22393C] bg-black/5 px-2 py-0.5 rounded">
              18.5204° N, 73.8567° E (MG Road)
            </span>
          </div>
        </div>

        {/* LIVE Status Indicator */}
        <div className="flex items-center space-x-2">
          <div className="px-3 py-1 rounded-lg bg-rose-600/10 border border-rose-500/30 text-rose-700 text-xs font-bold flex items-center space-x-1.5">
            <Radio className="w-3.5 h-3.5 text-rose-600 animate-ping" />
            <span className="tracking-widest">LIVE EDGE FEED</span>
          </div>
        </div>
      </div>

      {/* Camera Selection Tabs */}
      <div className="flex space-x-2 border-b border-[#46707E]/20 pb-3">
        {(['Front Camera', 'Left Camera', 'Right Camera', 'Rear Camera'] as const).map((cam) => (
          <button
            key={cam}
            onClick={() => setActiveCamera(cam)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
              activeCamera === cam
                ? 'bg-[#22393C] text-white shadow-md'
                : 'bg-white/60 text-[#46707E] hover:bg-white'
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
            <span>{cam}</span>
          </button>
        ))}
      </div>

      {/* Main Grid: Video Player + AI Pipeline Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Video Player with AI Overlay */}
        <div className="lg:col-span-2 space-y-4">
          <div className="relative rounded-2xl overflow-hidden shadow-2xl bg-black border border-[#46707E]/30 aspect-video">
            {videoSrc ? (
              /* Real uploaded video playback */
              <video
                ref={videoRef}
                src={videoSrc}
                className="w-full h-full object-cover"
                autoPlay
                loop
                muted
                playsInline
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
              />
            ) : imageSrc ? (
              /* Real uploaded image preview */
              <img
                src={imageSrc}
                alt="Uploaded Evidence"
                className="w-full h-full object-contain bg-[#111]"
              />
            ) : (
              /* Canvas simulated road video */
              <canvas
                ref={canvasRef}
                width={800}
                height={450}
                className="w-full h-full object-cover"
              />
            )}

            {/* Top Video Overlay Telemetry */}
            <div className="absolute top-4 left-4 right-4 flex items-center justify-between text-xs font-mono text-white/90 bg-black/40 backdrop-blur-md px-3.5 py-1.5 rounded-lg border border-white/10 pointer-events-none">
              <div className="flex items-center space-x-4">
                <span className="flex items-center gap-1 text-emerald-400 font-bold">
                  <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                  STREAM_01: {videoSrc ? (uploadedFileName?.toUpperCase() || 'UPLOADED_VIDEO') : activeCamera.toUpperCase()}
                </span>
                <span>YOLO — Edge Inference</span>
                <span>LATENCY: 14ms</span>
              </div>
              <div className="flex items-center space-x-3 text-white/70">
                <span>BITRATE: 4.2 Mbps</span>
                <span>{currentBus.speed}</span>
              </div>
            </div>

            {/* Bottom Video Controls Bar */}
            <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/90 via-black/50 to-transparent">
              {/* Progress Scrubber */}
              <div
                onClick={handleScrubberClick}
                className="w-full bg-white/20 h-1.5 rounded-full overflow-hidden cursor-pointer mb-3"
              >
                <div
                  className="bg-[#AFBB98] h-full rounded-full transition-all"
                  style={{
                    width: `${
                      videoSrc
                        ? Math.min(100, Math.max(0, (playbackTime / (videoDuration || 1)) * 100))
                        : (playbackTime / 3600) * 100
                    }%`,
                  }}
                ></div>
              </div>

              <div className="flex items-center justify-between text-white text-xs">
                <div className="flex items-center space-x-4">
                  <button
                    onClick={handlePlayPause}
                    className="p-1.5 rounded-lg bg-white/15 hover:bg-white/25 transition-colors"
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </button>
                  <span className="font-mono text-xs text-[#CECDB9]">
                    {formatTime(playbackTime)} / {videoSrc ? formatTime(videoDuration) : '01:00:00'}
                  </span>
                  <div className="hidden sm:flex items-center space-x-1.5 text-white/60">
                    <Volume2 className="w-3.5 h-3.5" />
                    <span>Live Audio Off</span>
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  <span className="px-2 py-0.5 rounded bg-white/10 text-[10px] font-mono font-semibold">
                    1080p @ 30fps
                  </span>
                  <button className="p-1 hover:text-[#AFBB98] transition-colors">
                    <Maximize2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Action Controls */}
          <div className="glass-panel p-4 rounded-xl flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*,image/*"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-4 py-2 rounded-xl bg-white border border-[#46707E]/30 text-[#22393C] text-xs font-bold flex items-center space-x-2 hover:bg-[#F6F7F3] transition-all shadow-sm"
              >
                <Upload className="w-3.5 h-3.5 text-[#46707E]" />
                <span>{uploadedFileName ? uploadedFileName : 'Upload Video'}</span>
              </button>
              <span className="text-xs text-gray-500 hidden sm:inline">
                Supports MP4, AVI, RTSP edge telemetry streams
              </span>
            </div>

            <button
              onClick={() => triggerEdgeProcessing(selectedBusId, uploadedFile, activeCamera)}
              disabled={pipelineState.isRunning}
              className={`px-5 py-2.5 rounded-xl text-xs font-bold flex items-center space-x-2 shadow-lg transition-all ${
                pipelineState.isRunning
                  ? 'bg-amber-600 text-white animate-pulse'
                  : 'bg-gradient-to-r from-[#22393C] to-[#46707E] text-white hover:opacity-95'
              }`}
            >
              <Sparkles className="w-4 h-4 text-[#AFBB98]" />
              <span>{pipelineState.isRunning ? 'Processing AI Pipeline...' : 'Start Processing'}</span>
            </button>
          </div>
        </div>

        {/* Right 1 Col: AI Pipeline Status & Live Detections */}
        <div className="space-y-6">
          {/* AI Pipeline Status Card */}
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-3">
              <h3 className="font-bold text-sm text-[#22393C] tracking-wide">
                AI Pipeline Status
              </h3>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-[#46707E]/10 text-[#46707E] font-semibold">
                Onboard Jetson Orin
              </span>
            </div>

            <div className="space-y-2.5">
              {pipelineState.stepLabels.map((label, index) => {
                const stepNum = index + 1;
                const isDone = pipelineState.currentStep >= stepNum;
                const isCurrent = pipelineState.isRunning && pipelineState.currentStep === stepNum;

                return (
                  <div
                    key={label}
                    className={`flex items-center justify-between p-2.5 rounded-xl text-xs font-medium transition-all ${
                      isCurrent
                        ? 'bg-amber-50 border border-amber-300 text-amber-900 font-bold'
                        : isDone
                        ? 'bg-emerald-50/60 border border-emerald-200/80 text-emerald-900'
                        : 'bg-black/5 text-gray-400'
                    }`}
                  >
                    <div className="flex items-center space-x-2.5">
                      <div
                        className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                          isDone
                            ? 'bg-emerald-600 text-white'
                            : isCurrent
                            ? 'bg-amber-500 text-white animate-spin'
                            : 'bg-gray-300 text-gray-600'
                        }`}
                      >
                        {isDone ? '✓' : stepNum}
                      </div>
                      <span>{label}</span>
                    </div>

                    <span className="text-[10px] font-mono">
                      {isCurrent ? 'RUNNING' : isDone ? 'ACTIVE' : 'IDLE'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* LIVE DETECTIONS Card */}
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-[#46707E]/20 pb-3">
              <h3 className="font-bold text-sm text-[#22393C] tracking-wide">
                LIVE DETECTIONS
              </h3>
              <div className="flex items-center space-x-2">
                {latestPedestrianRiskTelemetry && (
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded-md bg-amber-500/10 text-amber-700 font-bold border border-amber-500/20">
                    {latestPedestrianRiskTelemetry.riskLevel} Risk
                  </span>
                )}
                {latestTrafficTelemetry && (
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded-md bg-rose-500/10 text-rose-700 font-bold border border-rose-500/20">
                    {latestTrafficTelemetry.trafficDensity} Density
                  </span>
                )}
                {uniqueVehicleCounts && (
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded-md bg-emerald-500/10 text-emerald-700 font-bold border border-emerald-500/20">
                    {uniqueVehicleCounts.total} Unique
                  </span>
                )}
                <span className="text-xs font-mono text-[#46707E] font-bold">
                  {pipelineState.detections.length} Active
                </span>
              </div>
            </div>

            {latestPedestrianRiskTelemetry && (
              <div className="p-3 rounded-xl bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200/80 text-xs flex items-center justify-between">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-amber-600 animate-ping"></span>
                    <span className="font-extrabold text-amber-900">
                      Pedestrian Risk: {latestPedestrianRiskTelemetry.riskLevel}
                    </span>
                  </div>
                  <span className="text-gray-600 block text-[10px] mt-0.5 font-mono">
                    {latestPedestrianRiskTelemetry.pedestrianCount} Pedestrians ({latestPedestrianRiskTelemetry.inConflictZoneCount} in corridor) • Est. Risk: {latestPedestrianRiskTelemetry.riskScore}/100 (Image-Based Estimate)
                  </span>
                </div>
                <span className="px-2 py-1 rounded text-[10px] font-mono font-black bg-amber-600 text-white shadow-sm">
                  {latestPedestrianRiskTelemetry.riskScore}/100
                </span>
              </div>
            )}

            {latestTrafficTelemetry && (
              <div className="p-3 rounded-xl bg-gradient-to-r from-rose-50 to-amber-50 border border-rose-200/80 text-xs flex items-center justify-between">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-rose-600 animate-ping"></span>
                    <span className="font-extrabold text-rose-900">Traffic Congestion: {latestTrafficTelemetry.congestionLevel}</span>
                  </div>
                  <span className="text-gray-600 block text-[10px] mt-0.5 font-mono">
                    {latestTrafficTelemetry.vehicleCount} Vehicles • Cars: {latestTrafficTelemetry.vehicleMix?.Car || '42%'} • Trucks: {latestTrafficTelemetry.vehicleMix?.Truck || '33%'}
                  </span>
                </div>
                <span className="px-2 py-1 rounded text-[10px] font-mono font-black bg-rose-600 text-white shadow-sm">
                  {latestTrafficTelemetry.trafficDensity}
                </span>
              </div>
            )}

            <div className="space-y-2.5">
              {pipelineState.detections.map((det, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-3 rounded-xl bg-white/70 border border-[#46707E]/15 hover:border-[#46707E]/30 transition-all"
                >
                  <div className="flex items-center space-x-3">
                    <div
                      className={`p-2 rounded-lg ${
                        det.type === 'damage'
                          ? 'bg-rose-100 text-rose-700'
                          : det.type === 'pedestrian'
                          ? 'bg-amber-100 text-amber-700'
                          : det.type === 'infra' || det.type === 'infrastructure'
                          ? 'bg-amber-100 text-amber-700'
                          : 'bg-[#46707E]/15 text-[#46707E]'
                      }`}
                    >
                      {det.type === 'damage' || det.type === 'infra' || det.type === 'infrastructure' ? (
                        <AlertTriangle className="w-4 h-4" />
                      ) : det.type === 'pedestrian' || det.label.includes('Pedestrian') ? (
                        <User className="w-4 h-4" />
                      ) : det.label.includes('Truck') ? (
                        <Truck className="w-4 h-4" />
                      ) : det.label.includes('Bus') ? (
                        <Bus className="w-4 h-4" />
                      ) : det.label.includes('Two-wheeler') || det.label.includes('Motorcycle') ? (
                        <Bike className="w-4 h-4" />
                      ) : (
                        <Car className="w-4 h-4" />
                      )}
                    </div>
                    <div>
                      <div className="text-xs font-bold text-[#22393C]">{det.label}</div>
                      <div className="text-[10px] text-gray-500">Confidence Score</div>
                    </div>
                  </div>

                  <div className="text-right">
                    <div
                      className={`text-xs font-mono font-bold ${
                        det.type === 'damage'
                          ? 'text-rose-600'
                          : det.type === 'pedestrian' || det.type === 'infra' || det.type === 'infrastructure'
                          ? 'text-amber-600'
                          : 'text-[#46707E]'
                      }`}
                    >
                      {(det.conf * 100).toFixed(0)}%
                    </div>
                    <div className={`text-[10px] font-semibold ${det.type === 'pedestrian' ? 'text-amber-600' : (det.type === 'infra' || det.type === 'damage' ? 'text-amber-600' : 'text-emerald-600')}`}>
                      {det.type === 'pedestrian' ? 'Hazard' : (det.type === 'infra' || det.type === 'damage' ? 'Defect' : 'Tracked')}
                    </div>
                  </div>
                </div>
              ))}
            </div>

          </div>
        </div>
      </div>
    </div>
  );
};

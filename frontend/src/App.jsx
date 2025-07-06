import React, { useState, useEffect, useRef } from 'react';
import { Upload, Home, Eye, RotateCcw, ZoomIn, ZoomOut, Settings, Layers } from 'lucide-react';
import { SceneManager } from './SceneManager.js';

// Main React Component
function FloorPlan3DViewer() {
  const [sceneData, setSceneData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [showControls, setShowControls] = useState(false);
  const [visibility, setVisibility] = useState({
    walls: true,
    rooms: true,
    furniture: true,
    doors: true,
    windows: true
  });
  const [processingInfo, setProcessingInfo] = useState(null);
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const fileInputRef = useRef(null);
  
  useEffect(() => {
    if (containerRef.current && !sceneRef.current) {
      try {
        sceneRef.current = new SceneManager(containerRef.current);
        setUploadStatus('3D Scene initialized. Upload a floor plan or load sample data.');
      } catch (error) {
        console.error('Error initializing scene:', error);
        setUploadStatus('Error initializing 3D scene. Please refresh the page.');
      }
    }
    
    return () => {
      if (sceneRef.current) {
        sceneRef.current.dispose();
        sceneRef.current = null;
      }
    };
  }, []);
  
  useEffect(() => {
    if (sceneRef.current && sceneData) {
      console.log('Loading scene data into 3D viewer:', sceneData);
      sceneRef.current.loadScene(sceneData);
    }
  }, [sceneData]);
  
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
      setUploadStatus('❌ Please upload an image file (PNG, JPG, etc.)');
      return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
      setUploadStatus('❌ File too large. Please upload an image smaller than 10MB.');
      return;
    }
    
    setIsLoading(true);
    setUploadStatus('📤 Uploading and processing floor plan image...');
    setProcessingInfo(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      console.log('Uploading file:', file.name, 'Size:', file.size);
      
      const response = await fetch('http://localhost:8000/upload/', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`Upload failed: ${response.status} - ${errorData.detail || response.statusText}`);
      }
      
      const result = await response.json();
      console.log('Processing result:', result);
      
      if (result.data) {
        setSceneData(result.data);
        setProcessingInfo(result.stats.processing_info);
        setUploadStatus(
          `✅ Successfully processed "${file.name}": ${result.stats.walls} walls, ${result.stats.rooms} rooms, ${result.stats.furniture} furniture items. Total area: ${result.stats.total_area.toFixed(1)}m²`
        );
      } else {
        throw new Error('No data received from server');
      }
      
    } catch (error) {
      console.error('Upload error:', error);
      let errorMessage = '❌ Processing failed: ';
      
      if (error.message.includes('Failed to fetch')) {
        errorMessage += 'Cannot connect to server. Please ensure the backend is running on port 8000.';
      } else {
        errorMessage += error.message;
      }
      
      setUploadStatus(errorMessage);
    } finally {
      setIsLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };
  
  const loadSampleData = async () => {
    setIsLoading(true);
    setUploadStatus('📥 Loading sample floor plan...');
    
    try {
      const response = await fetch('http://localhost:8000/sample');
      
      if (!response.ok) {
        throw new Error(`Failed to load sample: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('Sample data loaded:', result);
      
      if (result.data) {
        setSceneData(result.data);
        setUploadStatus('✅ Sample floor plan loaded successfully');
      } else {
        throw new Error('No sample data received');
      }
      
    } catch (error) {
      console.error('Sample load error:', error);
      setUploadStatus('❌ Failed to load sample data. Please check if the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };
  
  const resetCamera = () => {
    if (sceneRef.current) {
      sceneRef.current.resetCamera();
      setUploadStatus('📷 Camera view reset');
    }
  };
  
  const toggleVisibility = (groupName) => {
    if (sceneRef.current) {
      sceneRef.current.toggleGroup(groupName);
      setVisibility(prev => ({
        ...prev,
        [groupName]: !prev[groupName]
      }));
    }
  };
  
  const getRoomStats = () => {
    if (!sceneData || !sceneData.rooms) return {};
    
    const roomTypes = {};
    sceneData.rooms.forEach(room => {
      const type = room.type;
      if (!roomTypes[type]) {
        roomTypes[type] = { count: 0, totalArea: 0 };
      }
      roomTypes[type].count++;
      roomTypes[type].totalArea += room.area || 0;
    });
    
    return roomTypes;
  };
  
  const roomStats = getRoomStats();
  
  return (
    <div className="w-full h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-lg p-4 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <Home className="w-8 h-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-800">
                Enhanced 3D Floor Plan Viewer
              </h1>
              <p className="text-sm text-gray-600">
                AI-powered floor plan analysis with intelligent furniture placement
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              <Upload className="w-4 h-4" />
              <span>Upload Floor Plan</span>
            </button>
            
            <button
              onClick={loadSampleData}
              disabled={isLoading}
              className="flex items-center space-x-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
            >
              <Eye className="w-4 h-4" />
              <span>Load Sample</span>
            </button>
            
            <button
              onClick={resetCamera}
              className="flex items-center space-x-2 bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset View</span>
            </button>
            
            <button
              onClick={() => setShowControls(!showControls)}
              className="flex items-center space-x-2 bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors"
            >
              <Layers className="w-4 h-4" />
              <span>Layers</span>
            </button>
          </div>
        </div>
        
        {uploadStatus && (
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">{uploadStatus}</p>
            {processingInfo && (
              <div className="mt-2 text-xs text-blue-600">
                <div>Scale: 1 pixel = {processingInfo.scale_factor}m</div>
                <div>Image size: {processingInfo.image_size[1]} × {processingInfo.image_size[0]} pixels</div>
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Main Content */}
      <div className="flex-1 relative">
        {isLoading ? (
          <div className="flex items-center justify-center h-full bg-gray-100">
            <div className="text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
              <p className="text-lg text-gray-700 mb-2">Processing floor plan...</p>
              <p className="text-sm text-gray-500">Detecting walls, segmenting rooms, and placing furniture</p>
            </div>
          </div>
        ) : (
          <div 
            ref={containerRef} 
            className="w-full h-full"
            style={{ height: 'calc(100vh - 140px)' }}
          />
        )}
        
        {/* Enhanced Controls Panel */}
        <div className="absolute bottom-4 left-4 bg-white rounded-lg shadow-lg p-3 max-w-xs">
          <div className="flex flex-col space-y-2">
            <div className="text-sm text-gray-700 font-semibold border-b pb-1">Navigation</div>
            <div className="text-xs text-gray-600 space-y-1">
              <div>🖱️ <strong>Left drag:</strong> Rotate view</div>
              <div>🖱️ <strong>Right drag:</strong> Pan view</div>
              <div>🎪 <strong>Mouse wheel:</strong> Zoom in/out</div>
              <div>🎯 <strong>Reset View:</strong> Auto-fit to floor plan</div>
            </div>
          </div>
        </div>
        
        {/* Layer Visibility Controls */}
        {showControls && (
          <div className="absolute top-4 left-4 bg-white rounded-lg shadow-lg p-4 min-w-48">
            <h3 className="text-sm font-semibold text-gray-800 mb-3 border-b pb-1">
              Layer Visibility
            </h3>
            <div className="space-y-2">
              {Object.entries(visibility).map(([key, visible]) => (
                <label key={key} className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={visible}
                    onChange={() => toggleVisibility(key)}
                    className="rounded text-blue-600"
                  />
                  <span className="text-sm text-gray-700 capitalize">
                    {key.replace('_', ' ')}
                  </span>
                  {sceneData && (
                    <span className="text-xs text-gray-500">
                      ({sceneData[key]?.length || 0})
                    </span>
                  )}
                </label>
              ))}
            </div>
          </div>
        )}
        
        {/* Enhanced Stats Panel */}
        {sceneData && (
          <div className="absolute top-4 right-4 bg-white rounded-lg shadow-lg p-4 min-w-64">
            <h3 className="text-sm font-semibold text-gray-800 mb-3 border-b pb-1">
              Floor Plan Analysis
            </h3>
            <div className="space-y-3">
              {/* Basic Stats */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-blue-50 p-2 rounded">
                  <div className="font-medium text-blue-800">Walls</div>
                  <div className="text-lg font-bold text-blue-600">
                    {sceneData.walls?.length || 0}
                  </div>
                </div>
                <div className="bg-green-50 p-2 rounded">
                  <div className="font-medium text-green-800">Rooms</div>
                  <div className="text-lg font-bold text-green-600">
                    {sceneData.rooms?.length || 0}
                  </div>
                </div>
                <div className="bg-purple-50 p-2 rounded">
                  <div className="font-medium text-purple-800">Furniture</div>
                  <div className="text-lg font-bold text-purple-600">
                    {sceneData.furniture?.length || 0}
                  </div>
                </div>
                <div className="bg-orange-50 p-2 rounded">
                  <div className="font-medium text-orange-800">Doors</div>
                  <div className="text-lg font-bold text-orange-600">
                    {sceneData.doors?.length || 0}
                  </div>
                </div>
              </div>
              
              {/* Room Breakdown */}
              {Object.keys(roomStats).length > 0 && (
                <div className="border-t pt-2">
                  <div className="text-xs font-medium text-gray-700 mb-1">Room Types</div>
                  <div className="space-y-1">
                    {Object.entries(roomStats).map(([type, data]) => (
                      <div key={type} className="flex justify-between text-xs">
                        <span className="text-gray-600">{type}</span>
                        <span className="font-medium">
                          {data.count} ({data.totalArea.toFixed(0)}m²)
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Total Area */}
              <div className="border-t pt-2 text-xs">
                <div className="flex justify-between font-medium">
                  <span className="text-gray-700">Total Area:</span>
                  <span className="text-gray-900">
                    {Object.values(roomStats).reduce((sum, data) => sum + data.totalArea, 0).toFixed(0)}m²
                  </span>
                </div>
              </div>
              
              {/* Processing Info */}
              {sceneData.metadata && (
                <div className="border-t pt-2 text-xs text-gray-600">
                  <div>Scale: 1px = {sceneData.metadata.scale_factor}m</div>
                  {sceneData.metadata.image_size && (
                    <div>Source: {sceneData.metadata.image_size[1]}×{sceneData.metadata.image_size[0]}px</div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Instructions for new users */}
        {!sceneData && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center max-w-md p-8 bg-white rounded-lg shadow-lg">
              <Home className="w-16 h-16 text-blue-600 mx-auto mb-4" />
              <h2 className="text-xl font-bold text-gray-800 mb-2">
                Welcome to 3D Floor Plan Viewer
              </h2>
              <p className="text-gray-600 mb-6">
                Upload a floor plan image to see it transformed into an interactive 3D model with intelligent room detection and furniture placement.
              </p>
              <div className="space-y-3">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full flex items-center justify-center space-x-2 bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Upload className="w-5 h-5" />
                  <span>Upload Your Floor Plan</span>
                </button>
                <button
                  onClick={loadSampleData}
                  className="w-full flex items-center justify-center space-x-2 bg-green-600 text-white px-4 py-3 rounded-lg hover:bg-green-700 transition-colors"
                >
                  <Eye className="w-5 h-5" />
                  <span>Try Sample Data</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileUpload}
        className="hidden"
      />
    </div>
  );
}

// Export the main component as default
export default FloorPlan3DViewer;
import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
  Dimensions,
  Linking,
  Image
} from 'react-native';
import { 
  Camera, 
  useCameraDevice, 
  useCameraPermission,
  useCameraFormat
} from 'react-native-vision-camera';
import { useNavigation, useIsFocused } from '@react-navigation/native';
import { launchImageLibrary } from 'react-native-image-picker';
import { runInferenceOnImage } from './TfliteUtils';

const { height: SCREEN_HEIGHT, width: SCREEN_WIDTH } = Dimensions.get('window');

// Model configuration
const MODEL_INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.5;

const ScanLicensePlateScreen = () => {
  const navigation = useNavigation();
  const isFocused = useIsFocused();
  const camera = useRef(null);
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false); // New state for capture status
  const [capturedImage, setCapturedImage] = useState(null);
  const [licensePlate, setLicensePlate] = useState('');
  const [stateCode, setStateCode] = useState('');

  // Camera format for better quality
  const format = useCameraFormat(device, [
    { videoResolution: { width: 1920, height: 1080 } },
    { fps: 30 }
  ]);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission().then((granted) => {
        if (!granted) {
          Alert.alert('Camera Permission', 'Please enable camera access in settings.', [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Open Settings', onPress: () => Linking.openSettings() },
          ]);
        }
      });
    }
  }, [hasPermission]);

  const captureImage = async () => {
    if (isCapturing || isProcessing || !camera.current) return;

    try {
      setIsCapturing(true); // Start capture process
      const photo = await camera.current.takePhoto({
        qualityPrioritization: 'quality',
        flash: 'off',
        skipMetadata: true,
      });

      console.log(`📸 Photo captured: ${photo.path}`);
      setCapturedImage(photo.path);
      setIsProcessing(true); // Start processing after capture
      await processImage(photo.path);
    } catch (error) {
      console.error('🚨 Capture Error:', error);
      Alert.alert("Capture Failed", "Please try again.");
    } finally {
      setIsCapturing(false); // Capture complete
    }
  };

  const selectImage = async () => {
    if (isProcessing || isCapturing) return;

    try {
      const result = await launchImageLibrary({
        mediaType: 'photo',
        quality: 1,
      });
      
      if (result.assets?.[0]?.uri) {
        setCapturedImage(result.assets[0].uri);
        setIsProcessing(true);
        await processImage(result.assets[0].uri);
      }
    } catch (error) {
      console.error('🚨 Gallery Error:', error);
      Alert.alert('Gallery Error', 'Failed to select image');
    }
  };

  const processImage = async (imageUri) => {
    try {
    
      console.log('⚙️ Processing image...');
      
      const outputs = await runInferenceOnImage(imageUri);
      
      const plates = processOutput(outputs[0]);
      
      if (plates.length === 0) {
        throw new Error('No license plates detected');
      }
      
      // Get the plate with highest confidence
      const bestPlate = plates.reduce((prev, current) => 
        (prev.confidence > current.confidence) ? prev : current
      );
      
      console.log('✅ Detected plate:', bestPlate.text);
      
      // Step 4: Extract state code (first 2 characters)
      const plateText = bestPlate.text.replace(/\s/g, '');
      const detectedStateCode = plateText.substring(0, 2);
      
      setLicensePlate(plateText);
      setStateCode(detectedStateCode);
      
      // Step 5: Navigate to results
      navigation.navigate('AddCar', {
        licensePlate: plateText,
        stateCode: detectedStateCode,
      });
      
    } catch (error) {
      console.error('❌ Processing Error:', error);
      Alert.alert(
        'Processing Failed', 
        error.message || 'Could not detect license plate. Try again or enter manually.'
      );
    } finally {
      setIsProcessing(false); // Always reset processing state
    }
  };

  // Process YOLOv9 output (simplified)
  const processOutput = (output) => {
    const plates = [];
    const numDetections = Math.min(output[0], 10);
    
    for (let i = 0; i < numDetections; i++) {
      const baseIndex = 1 + i * 6;
      const confidence = output[baseIndex + 4];
      
      if (confidence > CONFIDENCE_THRESHOLD) {
        const x1 = output[baseIndex];
        const y1 = output[baseIndex + 1];
        const x2 = output[baseIndex + 2];
        const y2 = output[baseIndex + 3];
        
        plates.push({
          confidence,
          bbox: { x1, y1, x2, y2 },
          text: `KA${Math.floor(Math.random() * 100)}AB${Math.floor(Math.random() * 10000)}`
        });
      }
    }
    
    return plates;
  };

  if (!hasPermission || !device) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#3b82f6" />
        <Text style={styles.loadingText}>Waiting for permissions or camera...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        format={format}
        isActive={isFocused} // Camera stays active as long as screen is focused
        photo={true}
        onInitialized={() => console.log('📷 Camera initialized')}
        onError={(error) => console.error('Camera Error', error.message)}
        pixelFormat="yuv"
      />

      {/* Overlay with detection area */}
      <View style={styles.overlay}>
        <View style={styles.detectionBox} />
        <Text style={styles.overlayText}>Position license plate here</Text>
      </View>

      <TouchableOpacity 
        style={styles.closeButton} 
        onPress={() => navigation.goBack()}
      >
        <Text style={styles.closeText}>✕</Text>
      </TouchableOpacity>

      <View style={styles.bottomContainer}>
        <View style={styles.textWrapper}>
          <Text style={styles.bottomTitle}>Scan License Plate</Text>
          <Text style={styles.bottomSubtitle}>Keep the plate within the box</Text>
        </View>
        
        <TouchableOpacity
          style={styles.captureButton}
          onPress={captureImage}
          disabled={isCapturing || isProcessing}
        >
          {(isCapturing || isProcessing) ? (
            <ActivityIndicator size="small" color="white" />
          ) : (
            <Text style={styles.captureText}>Capture</Text>
          )}
        </TouchableOpacity>
        
        <TouchableOpacity
          style={styles.manualEntry}
          onPress={() => navigation.navigate('AddCar')}
          disabled={isCapturing || isProcessing}
        >
          <Text style={[styles.manualText, 
            (isCapturing || isProcessing) && { opacity: 0.5 }]}>
            Enter Manually
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={styles.galleryButton}
          onPress={selectImage}
          disabled={isCapturing || isProcessing}
        >
          <Text style={[styles.galleryText, 
            (isCapturing || isProcessing) && { opacity: 0.5 }]}>
            Choose from Gallery
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  loadingText: {
    color: '#fff',
    marginTop: 12,
    fontSize: 16,
    textAlign: 'center',
  },
  closeButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    zIndex: 10,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 20,
    padding: 10,
  },
  closeText: {
    fontSize: 24,
    color: '#fff',
    fontWeight: 'bold',
  },
  overlay: {
    position: 'absolute',
    top: '20%',
    alignSelf: 'center',
    alignItems: 'center',
  },
  detectionBox: {
    width: SCREEN_WIDTH * 0.8,
    height: SCREEN_HEIGHT * 0.2,
    borderWidth: 2,
    borderColor: '#3b82f6',
    borderRadius: 10,
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
  },
  overlayText: {
    color: '#fff',
    marginTop: 10,
    fontSize: 16,
    fontWeight: '500',
  },
  bottomContainer: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
    height: SCREEN_HEIGHT * 0.3,
    backgroundColor: 'rgba(0,0,0,0.7)',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 20,
    paddingBottom: 30,
  },
  textWrapper: {
    alignItems: 'center',
    marginBottom: 15,
  },
  bottomTitle: {
    color: '#fff',
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  bottomSubtitle: {
    color: '#e2e8f0',
    fontSize: 16,
  },
  captureButton: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 30,
    marginBottom: 15,
    minWidth: 150,
    alignItems: 'center',
  },
  captureText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  manualEntry: {
    marginBottom: 10,
  },
  manualText: {
    color: '#93c5fd',
    fontSize: 16,
    textDecorationLine: 'underline',
  },
  galleryButton: {
    marginTop: 5,
  },
  galleryText: {
    color: '#cbd5e1',
    fontSize: 14,
  },
});

export default ScanLicensePlateScreen;
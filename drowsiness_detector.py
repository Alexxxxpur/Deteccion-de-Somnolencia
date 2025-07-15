"""
Sistema de Detección de Somnolencia - Lógica Principal
Optimizado con MediaPipe, alertas inmediatas y detección de cabeceos
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time
import threading
from typing import Tuple, Optional, Dict, Any
import pygame
from dataclasses import dataclass


@dataclass
class DrowsinessMetrics:
    """Métricas de somnolencia para tracking"""
    blinks_count: int = 0
    eye_closed_time: float = 0.0
    last_blink_time: float = 0.0
    drowsiness_level: str = "ALERTA"
    ear_left: float = 0.0
    ear_right: float = 0.0
    head_pose: Dict[str, float] = None
    is_face_detected: bool = False
    
    def __post_init__(self):
        if self.head_pose is None:
            self.head_pose = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}


class DrowsinessDetector:
    """
    Detector de somnolencia optimizado usando MediaPipe
    Detecta parpadeos, tiempo de ojos cerrados y posición de la cabeza
    """
    
    def __init__(self, 
                 ear_threshold: float = 0.25,
                 blink_threshold: int = 3,
                 drowsy_time_threshold: float = 1.5,
                 microsleep_threshold: float = 3.0):
        """
        Inicializa el detector de somnolencia
        
        Args:
            ear_threshold: Umbral de aspecto del ojo para detectar parpadeo
            blink_threshold: Número mínimo de parpadeos para validar vivacidad
            drowsy_time_threshold: Tiempo en segundos para detectar somnolencia
            microsleep_threshold: Tiempo en segundos para detectar microsueño
        """
        # Configuración de MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Inicializar Face Mesh con configuraciones optimizadas
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Índices de landmarks para los ojos (MediaPipe 468 puntos)
        self.LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        
        # Índices para cejas y parietales (detección de cabeceos como el código original)
        self.LEFT_EYEBROW = 70    # Ceja izquierda (x8 en código original)
        self.RIGHT_EYEBROW = 300  # Ceja derecha (x7 en código original)
        self.LEFT_PARIETAL = 368  # Parietal izquierdo (x6 en código original) 
        self.RIGHT_PARIETAL = 139 # Parietal derecho (x5 en código original)
        
        # Configuración de umbrales
        self.ear_threshold = ear_threshold
        self.blink_threshold = blink_threshold
        self.drowsy_time_threshold = drowsy_time_threshold
        self.microsleep_threshold = microsleep_threshold
        
        # Variables de estado
        self.metrics = DrowsinessMetrics()
        self.eye_closed_start_time = None
        self.is_blinking = False
        self.consecutive_drowsy_frames = 0
        self.alert_active = False
        
        # Calibración automática para cabeceos
        self.baseline_y_diff = None
        self.baseline_nose_y = None
        self.calibration_frames = 0
        self.is_calibrated = False
        
        # Sistema de audio
        self._init_audio_system()
        
        # Thread para alertas
        self.alert_thread = None
        self.stop_alert = threading.Event()
    
    def _init_audio_system(self):
        """Inicializa el sistema de audio para alertas"""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.audio_available = True
            print("✅ Audio inicializado correctamente")
        except pygame.error:
            print("⚠️ Warning: No se pudo inicializar el sistema de audio")
            self.audio_available = False
    
    def _calculate_ear(self, eye_landmarks: list, landmarks) -> float:
        """
        Calcula el Eye Aspect Ratio (EAR) para un ojo
        
        Args:
            eye_landmarks: Lista de índices de landmarks del ojo
            landmarks: Landmarks faciales de MediaPipe
            
        Returns:
            float: Valor EAR (0-1, donde valores bajos indican ojo cerrado)
        """
        try:
            # Obtener coordenadas de los puntos del ojo
            points = []
            for idx in eye_landmarks:
                landmark = landmarks.landmark[idx]
                points.append([landmark.x, landmark.y])
            
            points = np.array(points)
            
            # Calcular distancias verticales
            vertical_1 = np.linalg.norm(points[1] - points[5])
            vertical_2 = np.linalg.norm(points[2] - points[4])
            
            # Calcular distancia horizontal
            horizontal = np.linalg.norm(points[0] - points[3])
            
            # Calcular EAR
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
            
        except (IndexError, ZeroDivisionError):
            return 0.0
    
    def _detect_head_pose(self, landmarks) -> Dict[str, float]:
        """
        Detecta la posición de la cabeza usando landmarks faciales
        
        Returns:
            Dict con ángulos de pitch, yaw, roll
        """
        try:
            # Puntos de referencia para calcular pose
            nose_tip = landmarks.landmark[1]
            chin = landmarks.landmark[18]
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[362]
            left_mouth = landmarks.landmark[61]
            right_mouth = landmarks.landmark[291]
            
            # Calcular ángulos aproximados
            # Yaw (rotación horizontal)
            eye_center_x = (left_eye.x + right_eye.x) / 2
            mouth_center_x = (left_mouth.x + right_mouth.x) / 2
            yaw = (mouth_center_x - eye_center_x) * 100
            
            # Pitch (inclinación vertical)
            eye_center_y = (left_eye.y + right_eye.y) / 2
            pitch = (nose_tip.y - eye_center_y) * 100
            
            # Roll (inclinación lateral)
            eye_angle = math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
            roll = math.degrees(eye_angle)
            
            return {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll
            }
            
        except (IndexError, AttributeError):
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    
    def _calibrate_baseline(self, landmarks):
        """Calibra valores baseline durante los primeros frames"""
        if self.is_calibrated:
            return
            
        # Obtener puntos clave
        frente = landmarks.landmark[10]
        barbilla = landmarks.landmark[152]
        nariz = landmarks.landmark[1]
        
        # Calcular métricas actuales
        y_diff = abs(frente.y - barbilla.y)
        nose_y_relative = nariz.y - frente.y
        
        # Acumular valores para calibración
        if self.baseline_y_diff is None:
            self.baseline_y_diff = y_diff
            self.baseline_nose_y = nose_y_relative
        else:
            # Promedio móvil para estabilizar
            self.baseline_y_diff = (self.baseline_y_diff + y_diff) / 2
            self.baseline_nose_y = (self.baseline_nose_y + nose_y_relative) / 2
        
        self.calibration_frames += 1
        
        # Calibración completa después de 30 frames (~1 segundo)
        if self.calibration_frames >= 30:
            self.is_calibrated = True
            print(f"✅ Calibración completada:")
            print(f"   Baseline Y-Diff: {self.baseline_y_diff:.3f}")
            print(f"   Baseline Nose-Y: {self.baseline_nose_y:.3f}")
    
    def _check_head_nod_position(self, landmarks) -> bool:
        """
        Verifica la posición de cabeceo usando calibración automática
        
        Returns:
            bool: True si la cabeza está en posición normal (no cabeceando)
        """
        try:
            # Calibrar durante los primeros frames
            self._calibrate_baseline(landmarks)
            
            if not self.is_calibrated:
                return True  # Durante calibración, asumir posición normal
            
            # Obtener puntos clave
            frente = landmarks.landmark[10]
            barbilla = landmarks.landmark[152]
            nariz = landmarks.landmark[1]
            
            # Calcular métricas actuales
            current_y_diff = abs(frente.y - barbilla.y)
            current_nose_y = nariz.y - frente.y
            
            # Calcular desviaciones respecto al baseline
            y_diff_change = (self.baseline_y_diff - current_y_diff) / self.baseline_y_diff
            nose_y_change = (current_nose_y - self.baseline_nose_y) / abs(self.baseline_nose_y + 0.001)
            
            # Detección de cabeceo basada en cambios significativos
            # Si Y-diff disminuye mucho = cabeceo (frente y barbilla se acercan)
            y_nod = y_diff_change > 0.3  # 30% de reducción
            
            # Si nariz se mueve mucho hacia abajo respecto a baseline = cabeceo
            nose_nod = nose_y_change > 0.5  # 50% de cambio
            
            # Es cabeceo si cualquier criterio se cumple
            is_nodding = y_nod or nose_nod
            
            return not is_nodding  # Retorna True si NO está cabeceando
            
        except Exception:
            return True  # En caso de error, asumir posición normal
    
    def _play_alert_sound(self):
        """Reproduce sonido de alerta"""
        if not self.audio_available:
            return
            
        try:
            # Generar tono de alerta usando pygame
            duration = 0.5  # segundos
            sample_rate = 22050
            frequency = 800  # Hz
            
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            
            for i in range(frames):
                wave = 0.3 * np.sin(2 * np.pi * frequency * i / sample_rate)
                arr[i] = [wave, wave]
            
            arr = (arr * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(arr)
            sound.play()
            
        except Exception as e:
            print(f"Error reproduciendo alerta: {e}")
    
    def _start_alert_sequence(self):
        """Inicia secuencia de alertas en thread separado"""
        if self.alert_thread and self.alert_thread.is_alive():
            return
            
        self.stop_alert.clear()
        self.alert_thread = threading.Thread(target=self._alert_loop)
        self.alert_thread.daemon = True
        self.alert_thread.start()
    
    def _alert_loop(self):
        """Loop de alertas que se ejecuta en thread separado"""
        while not self.stop_alert.is_set():
            self._play_alert_sound()
            time.sleep(0.8)  # Pausa entre alertas
    
    def _stop_alert_sequence(self):
        """Detiene la secuencia de alertas"""
        self.stop_alert.set()
        if self.alert_thread:
            self.alert_thread.join(timeout=1.0)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DrowsinessMetrics]:
        """
        Procesa un frame de video y detecta signos de somnolencia
        
        Args:
            frame: Frame de video en formato BGR
            
        Returns:
            Tuple[np.ndarray, DrowsinessMetrics]: Frame procesado y métricas
        """
        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Reset métricas de frame
        self.metrics.is_face_detected = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.metrics.is_face_detected = True
                
                # Calcular EAR para ambos ojos
                self.metrics.ear_left = self._calculate_ear(self.LEFT_EYE_LANDMARKS, face_landmarks)
                self.metrics.ear_right = self._calculate_ear(self.RIGHT_EYE_LANDMARKS, face_landmarks)
                
                # Promedio de EAR
                avg_ear = (self.metrics.ear_left + self.metrics.ear_right) / 2.0
                
                # Verificar posición de cabeza (detección de cabeceos)
                head_nod_ok = self._check_head_nod_position(face_landmarks)
                
                current_time = time.time()
                
                # Detección de parpadeo (FUNCIONA SIEMPRE, incluso con cabeceo)
                if avg_ear < self.ear_threshold:
                    if not self.is_blinking:
                        self.is_blinking = True
                        self.eye_closed_start_time = current_time
                    else:
                        # Calcular tiempo con ojos cerrados
                        self.metrics.eye_closed_time = current_time - self.eye_closed_start_time
                        
                        # Determinar nivel de somnolencia
                        if self.metrics.eye_closed_time > self.microsleep_threshold:
                            self.metrics.drowsiness_level = "MICROSUEÑO"
                            # ALERTA INMEDIATA para microsueño
                            if not self.alert_active:
                                self.alert_active = True
                                self._start_alert_sequence()
                        elif self.metrics.eye_closed_time > self.drowsy_time_threshold:
                            self.metrics.drowsiness_level = "SOMNOLIENTO"
                            # ALERTA INMEDIATA para somnolencia (≥ 1.5s)
                            if not self.alert_active:
                                self.alert_active = True
                                self._start_alert_sequence()
                        else:
                            self.metrics.drowsiness_level = "NORMAL"
                
                else:
                    if self.is_blinking:
                        # Fin del parpadeo
                        self.is_blinking = False
                        # Solo incrementar contador si la cabeza está bien posicionada
                        if head_nod_ok:
                            self.metrics.blinks_count += 1
                        self.metrics.last_blink_time = current_time
                        self.metrics.eye_closed_time = 0.0
                        self.consecutive_drowsy_frames = 0
                        self.metrics.drowsiness_level = "ALERTA"
                        
                        # Detener alerta cuando termine el parpadeo
                        if self.alert_active:
                            self.alert_active = False
                            self._stop_alert_sequence()
                    
                    # Si la cabeza no está bien posicionada (cabeceo), resetear contador
                    if not head_nod_ok:
                        self.consecutive_drowsy_frames = 0
                        self.metrics.blinks_count = 0  # Reset como en código original
                        self.metrics.drowsiness_level = "CABECEO_DETECTADO"
                
                # Detectar posición de cabeza para métricas
                self.metrics.head_pose = self._detect_head_pose(face_landmarks)
                
                # Dibujar landmarks de ojos, cabeza y TODA la malla facial
                self._draw_eye_landmarks(frame, face_landmarks)
                self._draw_head_landmarks(frame, face_landmarks)
                self._draw_full_face_mesh(frame, face_landmarks)
        
        else:
            # No se detectó rostro
            self.metrics.drowsiness_level = "SIN_ROSTRO"
            if self.alert_active:
                self.alert_active = False
                self._stop_alert_sequence()
        
        return frame, self.metrics
    
    def _draw_eye_landmarks(self, frame: np.ndarray, landmarks):
        """Dibuja los landmarks de los ojos en el frame"""
        h, w = frame.shape[:2]
        
        # Dibujar puntos de los ojos
        for eye_landmarks in [self.LEFT_EYE_LANDMARKS, self.RIGHT_EYE_LANDMARKS]:
            points = []
            for idx in eye_landmarks:
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Conectar puntos del ojo
            points = np.array(points, np.int32)
            cv2.polylines(frame, [points], True, (255, 255, 0), 1)
    
    def _draw_head_landmarks(self, frame: np.ndarray, landmarks):
        """Dibuja información de detección de cabeceos (sin debug)"""
        h, w = frame.shape[:2]
        
        # Verificar estado de calibración
        if not self.is_calibrated:
            cv2.putText(frame, f"CALIBRANDO... {self.calibration_frames}/30", 
                       (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return
        
        # Verificar estado de cabeceo
        head_ok = self._check_head_nod_position(landmarks)
        
        # Mostrar solo el estado principal
        cabeceo_text = "Cabeza OK" if head_ok else "CABECEO!"
        cabeceo_color = (0, 255, 0) if head_ok else (0, 0, 255)
        cv2.putText(frame, cabeceo_text, (w - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, cabeceo_color, 2)
        
        # Dibujar puntos clave para referencia visual
        key_points = [
            (1, (0, 255, 255), "Nariz"),        # Punta de nariz ✓
            (152, (255, 0, 255), "Barbilla"),   # Barbilla central ✓ 
            (10, (255, 255, 0), "Frente")       # Frente central
        ]
        
        for idx, color, label in key_points:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.putText(frame, f"{label}", (x - 20, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Dibujar puntos clave para referencia visual
        key_points = [
            (1, (0, 255, 255), "Nariz"),        # Punta de nariz ✓
            (152, (255, 0, 255), "Barbilla"),   # Barbilla central ✓ 
            (10, (255, 255, 0), "Frente")       # Frente central
        ]
        
        for idx, color, label in key_points:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.putText(frame, f"{label}", (x - 20, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def _draw_full_face_mesh(self, frame: np.ndarray, landmarks):
        """Dibuja la malla facial completa como en el código original"""
        # Usar MediaPipe para dibujar TODA la malla facial
        self.mp_drawing.draw_landmarks(
            frame, 
            landmarks, 
            self.mp_face_mesh.FACEMESH_CONTOURS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
        )
    
    def reset_metrics(self):
        """Reinicia las métricas del detector"""
        self.metrics = DrowsinessMetrics()
        self.eye_closed_start_time = None
        self.is_blinking = False
        self.consecutive_drowsy_frames = 0
        if self.alert_active:
            self._stop_alert_sequence()
            self.alert_active = False
    
    def cleanup(self):
        """Limpia recursos del detector"""
        self._stop_alert_sequence()
        if self.audio_available:
            pygame.mixer.quit()


# Función utilitaria para crear instancia del detector
def create_drowsiness_detector(**kwargs) -> DrowsinessDetector:
    """
    Crea una instancia del detector de somnolencia con configuración personalizada
    
    Args:
        **kwargs: Parámetros de configuración del detector
        
    Returns:
        DrowsinessDetector: Instancia configurada del detector
    """
    return DrowsinessDetector(**kwargs)


if __name__ == "__main__":
    # Ejemplo de uso básico
    detector = create_drowsiness_detector()
    
    # Simular procesamiento con webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("🔍 Detector de Somnolencia Iniciado")
    print("👁️ Cierra los ojos >1.5s para probar alertas")
    print("🗣️ Cabecear reseteará el contador")
    print("❌ Presiona 'q' para salir")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Procesar frame
            processed_frame, metrics = detector.process_frame(frame)
            
            # Mostrar información
            cv2.putText(processed_frame, f"Parpadeos: {metrics.blinks_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Estado: {metrics.drowsiness_level}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar tiempo con ojos cerrados si aplica
            if metrics.eye_closed_time > 0:
                cv2.putText(processed_frame, f"Ojos cerrados: {metrics.eye_closed_time:.1f}s", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            
            # Mostrar EAR promedio
            avg_ear = (metrics.ear_left + metrics.ear_right) / 2.0
            cv2.putText(processed_frame, f"EAR: {avg_ear:.3f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Indicador visual de alerta
            if detector.alert_active:
                cv2.rectangle(processed_frame, (0, 0), (processed_frame.shape[1], processed_frame.shape[0]), (0, 0, 255), 8)
                cv2.putText(processed_frame, "!!! ALERTA DE SOMNOLENCIA !!!", 
                           (processed_frame.shape[1]//2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            cv2.imshow("🔍 Detector de Somnolencia - Presiona 'q' para salir", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        print("✅ Detector cerrado correctamente")
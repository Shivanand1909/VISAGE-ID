from flask import Flask, render_template, Response, jsonify, request, session
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
from werkzeug.utils import secure_filename
import time
import threading
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'visage_id_secret_key_2025'  # For session management

class FaceRecognitionSystem:
    def __init__(self):
        self.images = []
        self.classNames = []
        self.encodeListKnown = []
        self.camera = None
        self.is_running = False
        self.frame_count = 0
        self.lock = threading.Lock()
        self.recognition_buffer = {}
        self.session_marks = {}
        self.load_known_faces()
        self.ensure_students_file()
    
    def ensure_students_file(self):
        """Ensure students.json exists"""
        if not os.path.exists('students.json'):
            with open('students.json', 'w') as f:
                json.dump({}, f)
    
    def register_student(self, name, roll_number):
        """Register a new student"""
        try:
            # Load existing students
            with open('students.json', 'r') as f:
                students = json.load(f)
            
            # Check if already registered
            if name in students:
                return False, "Student already registered"
            
            # Add new student
            students[name] = {
                'roll_number': roll_number,
                'registered_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_classes': 0,
                'attended_classes': 0
            }
            
            # Save
            with open('students.json', 'w') as f:
                json.dump(students, f, indent=4)
            
            print(f'✅ Registered: {name} ({roll_number})')
            return True, "Student registered successfully"
        except Exception as e:
            print(f'❌ Registration error: {e}')
            return False, str(e)
    
    def get_student_info(self, name):
        """Get student information"""
        try:
            with open('students.json', 'r') as f:
                students = json.load(f)
            return students.get(name)
        except:
            return None
    
    def calculate_attendance_percentage(self, name):
        """Calculate attendance percentage for a student"""
        try:
            with open('students.json', 'r') as f:
                students = json.load(f)
            
            if name not in students:
                return None
            
            student = students[name]
            total = student.get('total_classes', 0)
            attended = student.get('attended_classes', 0)
            
            if total == 0:
                return 0
            
            percentage = (attended / total) * 100
            return round(percentage, 2)
        except:
            return None
    
    def update_total_classes(self, class_count=1):
        """Update total classes for all students"""
        try:
            with open('students.json', 'r') as f:
                students = json.load(f)
            
            for name in students:
                students[name]['total_classes'] = students[name].get('total_classes', 0) + class_count
            
            with open('students.json', 'w') as f:
                json.dump(students, f, indent=4)
            
            print(f'✅ Updated total classes (+{class_count}) for all students')
            return True
        except Exception as e:
            print(f'❌ Error updating classes: {e}')
            return False
    
    def load_known_faces(self):
        """Load known faces from images folder"""
        path = 'images'
        if not os.path.exists(path):
            os.makedirs(path)
            return
        
        self.images = []
        self.classNames = []
        
        print(f'\n🔍 Loading faces from: {os.path.abspath(path)}')
        
        for person in os.listdir(path):
            person_folder = os.path.join(path, person)
            if os.path.isdir(person_folder):
                for file in os.listdir(person_folder):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_folder, file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            self.images.append(img)
                            self.classNames.append(person)
        
        if len(self.images) > 0:
            self.encodeListKnown = self.findEncodings(self.images)
            print(f'✅ Loaded {len(self.encodeListKnown)} encodings for {len(set(self.classNames))} people')
            print(f'👥 People: {list(set(self.classNames))}')
        else:
            print('⚠️ No faces found')
    
    def findEncodings(self, images):
        """Encode known faces"""
        encodeList = []
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img_rgb)
            if encodings:
                encodeList.append(encodings[0])
        return encodeList
    
    def markAttendance(self, name):
        """Mark attendance and update student records"""
        with self.lock:
            # Check if already marked in this session
            if name in self.session_marks:
                print(f'ℹ️  {name} already marked in this session')
                return False
            
            now = datetime.now()
            dateString = now.strftime('%Y-%m-%d')
            timeString = now.strftime('%H:%M:%S')
            
            print(f'\n{"="*60}')
            print(f'📝 MARKING ATTENDANCE')
            print(f'   Name: {name}')
            print(f'   Date: {dateString}')
            print(f'   Time: {timeString}')
            
            try:
                # Ensure CSV exists
                if not os.path.exists('attendance.csv'):
                    with open('attendance.csv', 'w', encoding='utf-8') as f:
                        f.write('Name,Time,Date\n')
                
                # Write attendance
                with open('attendance.csv', 'a', encoding='utf-8') as f:
                    f.write(f'{name},{timeString},{dateString}\n')
                    f.flush()
                    os.fsync(f.fileno())
                
                # Update student attendance count
                with open('students.json', 'r') as f:
                    students = json.load(f)
                
                if name in students:
                    students[name]['attended_classes'] = students[name].get('attended_classes', 0) + 1
                    
                    with open('students.json', 'w') as f:
                        json.dump(students, f, indent=4)
                    
                    # Calculate percentage
                    percentage = self.calculate_attendance_percentage(name)
                    print(f'   📊 Attendance: {percentage}%')
                    
                    if percentage and percentage < 75:
                        print(f'   ⚠️ WARNING: {name} has low attendance ({percentage}%)')
                
                # Mark as done in session
                self.session_marks[name] = True
                
                print(f'   ✅ SUCCESS! Attendance marked')
                print(f'{"="*60}\n')
                
                return True
                
            except Exception as e:
                print(f'   ❌ ERROR: {e}')
                print(f'{"="*60}\n')
                return False
    
    def generate_frames(self):
        """Generate video frames"""
        print('🎥 Frame generation started')
        
        while self.is_running:
            if not self.camera or not self.camera.isOpened():
                break
                
            success, frame = self.camera.read()
            if not success:
                break
            
            self.frame_count += 1
            current_time = time.time()
            
            if self.frame_count % 10 == 0:
                small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    
                    if len(self.encodeListKnown) == 0:
                        y1, x2, y2, x1 = face_location
                        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, "NO FACES LOADED", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        continue
                    
                    matches = face_recognition.compare_faces(self.encodeListKnown, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(self.encodeListKnown, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                            name = self.classNames[best_match_index].upper()
                            distance = face_distances[best_match_index]
                            
                            already_marked = name in self.session_marks
                            
                            if not already_marked:
                                if name not in self.recognition_buffer or (current_time - self.recognition_buffer[name]) > 2:
                                    self.recognition_buffer[name] = current_time
                                    print(f'🎯 RECOGNIZED: {name} (distance: {distance:.3f})')
                                    self.markAttendance(name)
                            
                            y1, x2, y2, x1 = face_location
                            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                            
                            if name in self.session_marks:
                                color = (0, 255, 0)
                                status = "MARKED"
                            else:
                                color = (0, 255, 255)
                                status = "MARKING..."
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            cv2.rectangle(frame, (x1, y2-40), (x2, y2), color, cv2.FILLED)
                            cv2.putText(frame, f"{name}", (x1+6, y2-22), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            cv2.putText(frame, status, (x1+6, y2-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        else:
                            y1, x2, y2, x1 = face_location
                            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "UNKNOWN", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def start_camera(self):
        """Start camera"""
        print(f'\n{"="*60}')
        print('🎥 STARTING NEW SESSION')
        
        if self.camera:
            self.camera.release()
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print('❌ Camera failed to open')
            return False
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.frame_count = 0
        self.recognition_buffer = {}
        self.session_marks = {}
        
        print('✅ Camera started')
        print(f'{"="*60}\n')
        return True
    
    def stop_camera(self):
        """Stop camera"""
        print('\n🛑 STOPPING SESSION')
        if self.session_marks:
            print(f'📊 Marked this session:')
            for name in self.session_marks.keys():
                print(f'   ✓ {name}')
        
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None

face_system = FaceRecognitionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_student', methods=['POST'])
def register_student():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        roll_number = data.get('roll_number', '').strip()
        
        if not name or not roll_number:
            return jsonify({'status': 'error', 'message': 'Name and roll number required'})
        
        success, message = face_system.register_student(name, roll_number)
        
        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_all_students')
def get_all_students():
    try:
        with open('students.json', 'r') as f:
            students = json.load(f)
        
        # Calculate percentages
        student_list = []
        for name, info in students.items():
            total = info.get('total_classes', 0)
            attended = info.get('attended_classes', 0)
            percentage = (attended / total * 100) if total > 0 else 0
            
            student_list.append({
                'name': name,
                'roll_number': info.get('roll_number', 'N/A'),
                'total_classes': total,
                'attended_classes': attended,
                'percentage': round(percentage, 2),
                'low_attendance': percentage < 75
            })
        
        return jsonify({'status': 'success', 'students': student_list})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/update_total_classes', methods=['POST'])
def update_total_classes():
    try:
        data = request.get_json()
        count = data.get('count', 1)
        
        success = face_system.update_total_classes(count)
        
        if success:
            return jsonify({'status': 'success', 'message': f'Added {count} class(es)'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to update'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/video_feed')
def video_feed():
    return Response(face_system.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    try:
        if face_system.start_camera():
            return jsonify({'status': 'success', 'message': 'Started new session'})
        return jsonify({'status': 'error', 'message': 'Camera failed'})
    except Exception as e:
        print(f'❌ Error: {e}')
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    try:
        face_system.stop_camera()
        return jsonify({'status': 'success', 'message': 'Session stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_attendance')
def get_attendance():
    try:
        if os.path.exists('attendance.csv'):
            df = pd.read_csv('attendance.csv', encoding='utf-8')
            today = datetime.now().strftime('%Y-%m-%d')
            today_attendance = df[df['Date'] == today].to_dict('records')
            
            return jsonify({
                'status': 'success',
                'attendance': today_attendance,
                'total_today': len(today_attendance)
            })
        return jsonify({'status': 'success', 'attendance': [], 'total_today': 0})
    except Exception as e:
        print(f'Error loading attendance: {e}')
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/reload_faces', methods=['POST'])
def reload_faces():
    try:
        face_system.load_known_faces()
        return jsonify({
            'status': 'success',
            'message': f'Loaded {len(face_system.encodeListKnown)} faces',
            'total_faces': len(face_system.encodeListKnown)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_known_faces')
def get_known_faces():
    try:
        unique = list(set(face_system.classNames))
        return jsonify({'status': 'success', 'faces': unique, 'total': len(unique)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/upload_face', methods=['POST'])
def upload_face():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file'})
        
        file = request.files['file']
        person_name = request.form.get('person_name', '').strip()
        
        if not person_name or file.filename == '':
            return jsonify({'status': 'error', 'message': 'Name and file required'})
        
        # Check if student is registered
        student_info = face_system.get_student_info(person_name)
        if not student_info:
            return jsonify({'status': 'error', 'message': 'Student not registered. Please register first.'})
        
        person_folder = os.path.join(app.config['UPLOAD_FOLDER'], person_name)
        os.makedirs(person_folder, exist_ok=True)
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(person_folder, f"{timestamp}_{filename}")
        file.save(filepath)
        
        img = cv2.imread(filepath)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        
        if len(faces) == 0:
            os.remove(filepath)
            return jsonify({'status': 'error', 'message': 'No face detected'})
        
        if len(faces) > 1:
            os.remove(filepath)
            return jsonify({'status': 'error', 'message': 'Multiple faces detected'})
        
        face_system.load_known_faces()
        return jsonify({
            'status': 'success',
            'message': f'Uploaded {person_name}',
            'total_faces': len(face_system.encodeListKnown)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/delete_person', methods=['POST'])
def delete_person():
    try:
        data = request.get_json()
        person_name = data.get('person_name', '').strip()
        
        if not person_name:
            return jsonify({'status': 'error', 'message': 'Name required'})
        
        person_folder = os.path.join(app.config['UPLOAD_FOLDER'], person_name)
        
        if not os.path.exists(person_folder):
            return jsonify({'status': 'error', 'message': 'Not found'})
        
        import shutil
        shutil.rmtree(person_folder)
        face_system.load_known_faces()
        
        return jsonify({
            'status': 'success',
            'message': f'Deleted {person_name}',
            'total_faces': len(face_system.encodeListKnown)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/reset_attendance', methods=['POST'])
def reset_attendance():
    try:
        if os.path.exists('attendance.csv'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f'attendance_backup_{timestamp}.csv'
            os.rename('attendance.csv', backup_name)
            print(f'📦 Backed up to {backup_name}')
        
        with open('attendance.csv', 'w', encoding='utf-8') as f:
            f.write('Name,Time,Date\n')
        
        print('✅ Created fresh attendance.csv')
        
        return jsonify({
            'status': 'success',
            'message': 'Created fresh attendance file'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print('\n' + '='*60)
    print('🚀 VISAGE ID - ATTENDANCE SYSTEM')
    print('📝 With Student Registration & Attendance Tracking')
    print('='*60)
    print('📡 http://127.0.0.1:5000')
    print('='*60 + '\n')
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
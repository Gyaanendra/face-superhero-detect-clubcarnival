import streamlit as st
import cv2
import numpy as np
import random
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import time

# List of superheroes and villains with descriptions and colors (50 characters)
characters = {
    # Heroes
    "Superman": {"description": "The Man of Steel - Protector of Metropolis", "color": (0, 0, 255)},
    "Batman": {"description": "The Dark Knight - Gotham's Guardian", "color": (30, 30, 30)},
    "Wonder Woman": {"description": "Amazon Princess - Warrior of Peace", "color": (255, 20, 147)},
    "Spider-Man": {"description": "Friendly Neighborhood Hero - Web-slinger", "color": (255, 0, 0)},
    "Iron Man": {"description": "Genius Billionaire - Armored Avenger", "color": (255, 165, 0)},
    "Captain America": {"description": "First Avenger - Symbol of Justice", "color": (0, 123, 255)},
    "The Flash": {"description": "Fastest Man Alive - Speed Force Champion", "color": (255, 215, 0)},
    "Green Lantern": {"description": "Space Cop - Willpower Incarnate", "color": (0, 255, 0)},
    "Aquaman": {"description": "King of Atlantis - Ocean's Protector", "color": (0, 191, 255)},
    "Thor": {"description": "God of Thunder - Asgardian Prince", "color": (255, 69, 0)},
    "Hulk": {"description": "Incredible Green Giant - Gamma-Powered Beast", "color": (34, 139, 34)},
    "Black Widow": {"description": "Master Spy - Deadly Assassin Turned Hero", "color": (139, 0, 0)},
    "Hawkeye": {"description": "Master Archer - Never Misses His Mark", "color": (128, 0, 128)},
    "Captain Marvel": {"description": "Cosmic Powerhouse - Binary Energy Master", "color": (255, 140, 0)},
    "Doctor Strange": {"description": "Master of Mystic Arts - Sorcerer Supreme", "color": (148, 0, 211)},
    "Black Panther": {"description": "Wakandan King - Vibranium Protector", "color": (75, 0, 130)},
    "Ant-Man": {"description": "Size-Changing Hero - Quantum Realm Explorer", "color": (255, 20, 147)},
    "Daredevil": {"description": "Man Without Fear - Hell's Kitchen Guardian", "color": (220, 20, 60)},
    "Wolverine": {"description": "Adamantium Claws - Immortal Mutant Warrior", "color": (255, 215, 0)},
    "Deadpool": {"description": "Merc with a Mouth - Regenerating Anti-Hero", "color": (255, 0, 0)},
    "Green Arrow": {"description": "Emerald Archer - Star City's Protector", "color": (0, 128, 0)},
    "Supergirl": {"description": "Maid of Might - Kryptonian Survivor", "color": (30, 144, 255)},
    "Shazam": {"description": "Champion of Magic - Ancient Wizard's Power", "color": (255, 215, 0)},
    "Nightwing": {"description": "Acrobatic Hero - Former Boy Wonder", "color": (0, 0, 139)},
    "Storm": {"description": "Weather Goddess - Mistress of Elements", "color": (255, 255, 255)},
    
    # Villains
    "Joker": {"description": "Clown Prince of Crime - Agent of Chaos", "color": (128, 0, 128)},
    "Loki": {"description": "God of Mischief - Trickster of Asgard", "color": (0, 100, 0)},
    "Thanos": {"description": "The Mad Titan - Universal Balancer", "color": (139, 69, 19)},
    "Magneto": {"description": "Master of Magnetism - Mutant Revolutionary", "color": (128, 128, 128)},
    "Green Goblin": {"description": "Pumpkin Bomb Maniac - Spider-Man's Nemesis", "color": (50, 205, 50)},
    "Lex Luthor": {"description": "Criminal Mastermind - Superman's Greatest Foe", "color": (0, 128, 0)},
    "Red Skull": {"description": "Nazi Super Soldier - Captain America's Enemy", "color": (178, 34, 34)},
    "Doctor Doom": {"description": "Armored Tyrant - Latverian Dictator", "color": (105, 105, 105)},
    "Venom": {"description": "Symbiote Menace - Dark Spider Reflection", "color": (0, 0, 0)},
    "Ultron": {"description": "AI Overlord - Humanity's Digital Threat", "color": (192, 192, 192)},
    "Darkseid": {"description": "Lord of Apokolips - Omega Beam Master", "color": (72, 61, 139)},
    "Harley Quinn": {"description": "Clown Princess - Joker's Chaotic Partner", "color": (255, 20, 147)},
    "Catwoman": {"description": "Master Thief - Feline Fatale", "color": (128, 0, 128)},
    "Two-Face": {"description": "Duality Incarnate - Coin-Flipping Criminal", "color": (169, 169, 169)},
    "Penguin": {"description": "Crime Boss - Gotham's Underworld Emperor", "color": (0, 0, 139)},
    "Riddler": {"description": "Master of Puzzles - Question Mark Killer", "color": (0, 128, 0)},
    "Scarecrow": {"description": "Master of Fear - Phobia Incarnate", "color": (139, 69, 19)},
    "Poison Ivy": {"description": "Plant Mistress - Eco-Terrorist Queen", "color": (34, 139, 34)},
    "Bane": {"description": "Venom-Enhanced Brute - Batman's Back-Breaker", "color": (105, 105, 105)},
    "Killer Croc": {"description": "Reptilian Monster - Sewer Savage", "color": (107, 142, 35)},
    "Mr. Freeze": {"description": "Ice Cold Killer - Cryogenic Criminal", "color": (173, 216, 230)},
    "Deathstroke": {"description": "Super Soldier Assassin - Tactical Terminator", "color": (255, 140, 0)},
    "Reverse Flash": {"description": "Speed Force Nemesis - Timeline Destroyer", "color": (255, 255, 0)},
    "Sinestro": {"description": "Yellow Lantern - Fear's Champion", "color": (255, 215, 0)},
    "Brainiac": {"description": "Super Intelligence - World Collector", "color": (0, 255, 127)}
}

@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 model for face detection"""
    try:
        model = YOLO('yolov8n.pt')  # Nano model for speed
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def detect_faces(image):
    """Detect faces using YOLO with improved face-centered bounding boxes"""
    model = load_yolo_model()
    if model is None:
        return []
    
    # Convert PIL to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run YOLO inference
    try:
        results = model(opencv_image, verbose=False)
    except Exception as e:
        st.error(f"YOLO inference failed: {e}")
        return []
    
    # Extract face detections (class 0 is person in COCO, we'll filter for faces)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Filter for person class (0) with high confidence
                if cls == 0 and conf > 0.4:  # Lowered threshold slightly for better detection
                    # Better face-centered bounding box calculation
                    person_height = y2 - y1
                    person_width = x2 - x1
                    
                    # Estimate face region (top 25% of person, centered horizontally)
                    face_height = int(person_height * 0.4)      # 40% of person height
                    face_width = int(min(person_width * 0.9, face_height * 1.4))   # Maintain aspect ratio
                    
                    # Position face box in upper portion but not at very top
                    face_y1 = int(y1 + person_height * 0.25)  # Start 5% down from top
                    face_y2 = face_y1 + face_height
                    
                    # Center horizontally
                    face_center_x = int(x1 + person_width / 2)
                    face_x1 = int(face_center_x - face_width / 2)
                    face_x2 = int(face_center_x + face_width / 2)
                    
                    # Ensure bounds are within image
                    h, w = opencv_image.shape[:2]
                    face_y1 = max(0, face_y1)
                    face_y2 = min(h, face_y2)
                    face_x1 = max(0, face_x1)
                    face_x2 = min(w, face_x2)
                    
                    # Only add if the box has valid dimensions
                    if face_y2 > face_y1 and face_x2 > face_x1 and (face_y2 - face_y1) > 20 and (face_x2 - face_x1) > 20:
                        face_crop = opencv_image[face_y1:face_y2, face_x1:face_x2]
                        detections.append({
                            'bbox': (face_x1, face_y1, face_x2, face_y2),
                            'confidence': conf,
                            'crop': face_crop
                        })
    
    # Sort detections by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections

def draw_detections(image, detections, characters_assigned):
    """Draw bounding boxes around detected faces with character assignments"""
    draw = ImageDraw.Draw(image)
    try:
        # Try loading a common system font
        font = ImageFont.truetype("Arial.ttf", 16) if os.name == 'nt' else ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Define colors for multiple faces
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        conf = detection['confidence']
        character = characters_assigned[i] if i < len(characters_assigned) else "Unknown"
        
        # Use different colors for different faces
        box_color = colors[i % len(colors)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        
        # Draw character name and confidence
        label = f"{character} ({conf:.1%})"
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        label_height = bbox[3] - bbox[1]
        
        # Position label above bounding box
        label_y = max(0, y1 - label_height - 5)
        draw.rectangle([x1, label_y, x1 + label_width + 4, label_y + label_height + 4], 
                      fill=box_color, outline=box_color)
        draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255), font=font)
    
    return image

def load_placeholder_image():
    """Load a placeholder image if webcam is unavailable"""
    # Create a simple placeholder image with text
    placeholder = Image.new('RGB', (320, 240), color=(100, 100, 100))
    draw = ImageDraw.Draw(placeholder)
    try:
        font = ImageFont.truetype("Arial.ttf", 20) if os.name == 'nt' else ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((160, 120), "Click 'Scan Face'\nto start", fill=(255, 255, 255), font=font, anchor="mm")
    return placeholder

def capture_with_live_preview(cap, duration=2, placeholder=None):
    """Capture frames for specified duration with live preview and return the best one"""
    frames = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            frames.append(image)
            
            # Show live preview if placeholder is provided
            if placeholder is not None:
                # Resize for display
                display_image = image.copy()
                display_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
                placeholder.image(display_image, caption="🔴 RECORDING...", use_container_width=True)
        
        time.sleep(0.1)  # Capture ~10 frames per second
    
    # Return the middle frame (most likely to be stable)
    if frames:
        return frames[len(frames)//2]
    return None

def map_to_characters(num_faces):
    """Randomly map to characters based on number of faces detected"""
    if num_faces == 0:
        return []
    
    # Get random characters without replacement
    available_characters = list(characters.keys())
    selected_characters = random.sample(available_characters, min(num_faces, len(available_characters)))
    return selected_characters

def display_character_info(character_list):
    """Display character information for multiple characters"""
    if not character_list:
        st.info("Click 'Scan Face' to discover your superhero identity!")
        return
        
    if len(character_list) == 1:
        st.markdown("### 🦸‍♂️ Your Superhero Identity:")
    else:
        st.markdown(f"### 🦸‍♂️ Your Superhero Team ({len(character_list)} members):")
    
    # Define colors for multiple faces (same as bounding box colors)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, character in enumerate(character_list):
        char_info = characters[character]
        # Use bounding box color instead of character color
        box_color = colors[i % len(colors)]
        
        st.markdown(
            f"""
            <div style="background-color: rgba({box_color[0]}, {box_color[1]}, {box_color[2]}, 0.15); 
                       padding: 15px; border-radius: 8px; border: 2px solid rgb({box_color[0]}, {box_color[1]}, {box_color[2]});
                       margin-bottom: 10px;">
                <h4 style="color: white; margin: 0; text-align: center;">
                    {f"Face {i+1}: " if len(character_list) > 1 else ""}{character}
                </h4>
                <p style="margin: 5px 0; text-align: center; font-style: italic; color: white;">
                    {char_info['description']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    st.set_page_config(page_title="Superhero Face Scanner", page_icon="🦸", layout="wide")
    
    # Compact header
    st.title("🦹‍♂️ Superhero Face Scanner")
    
    # Initialize session state
    if 'scanned_image' not in st.session_state:
        st.session_state.scanned_image = None
    if 'characters' not in st.session_state:
        st.session_state.characters = []
    if 'detections' not in st.session_state:
        st.session_state.detections = []
    
    # Main layout in columns
    col1, col2, col3 = st.columns([2, 1, 2])
    
    # Left column - Camera feed
    with col1:
        st.subheader("📸 Camera")
        
        # Create camera display placeholder
        camera_placeholder = st.empty()
        
        # Show current image (scanned or placeholder)
        if st.session_state.scanned_image is not None:
            # Resize image to fit better
            img_display = st.session_state.scanned_image.copy()
            img_display.thumbnail((400, 300), Image.Resampling.LANCZOS)
            camera_placeholder.image(img_display, caption="Last Scan", use_container_width=True)
        else:
            placeholder_img = load_placeholder_image()
            camera_placeholder.image(placeholder_img, caption="Ready to scan", use_container_width=True)
    
    # Middle column - Controls
    with col2:
        st.subheader("🎯 Controls")
        
        if st.button("🔍 Scan Face", type="primary", use_container_width=True):
            # Create placeholders for countdown
            countdown_placeholder = st.empty()
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam")
                st.session_state.scanned_image = load_placeholder_image()
                st.session_state.characters = []
                st.session_state.detections = []
            else:
                try:
                    # Show live preview during countdown
                    for i in range(3, 0, -1):
                        countdown_placeholder.info(f"📸 Get ready! Scanning in {i}...")
                        
                        # Show live camera feed during countdown
                        ret, frame = cap.read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            preview_image = Image.fromarray(frame_rgb)
                            preview_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
                            camera_placeholder.image(preview_image, caption=f"Get Ready! {i}...", use_container_width=True)
                        
                        time.sleep(1)
                    
                    countdown_placeholder.success("📹 Recording for 2 seconds...")
                    
                    # Capture frames for 2 seconds with live preview
                    image = capture_with_live_preview(cap, duration=2, placeholder=camera_placeholder)
                    
                    countdown_placeholder.empty()  # Clear countdown
                    
                    if image is None:
                        st.error("Failed to capture frames")
                        st.session_state.scanned_image = load_placeholder_image()
                        st.session_state.characters = []
                        st.session_state.detections = []
                        placeholder_img = load_placeholder_image()
                        camera_placeholder.image(placeholder_img, caption="Ready to scan", use_container_width=True)
                    else:
                        with st.spinner("Analyzing faces..."):
                            # Perform face detection
                            detections = detect_faces(image)
                            num_faces = len(detections)
                            
                            if num_faces > 0:
                                # Assign characters to each detected face
                                assigned_characters = map_to_characters(num_faces)
                                annotated_image = draw_detections(image.copy(), detections, assigned_characters)
                                
                                # Store results
                                st.session_state.scanned_image = annotated_image
                                st.session_state.characters = assigned_characters
                                st.session_state.detections = detections
                                
                                # Show final result
                                final_display = annotated_image.copy()
                                final_display.thumbnail((400, 300), Image.Resampling.LANCZOS)
                                camera_placeholder.image(final_display, caption="✨ Scan Complete!", use_container_width=True)
                                
                                if num_faces == 1:
                                    st.success(f"✨ You are {assigned_characters[0]}!")
                                else:
                                    st.success(f"✨ {num_faces} faces detected! Your team is ready!")
                            else:
                                st.session_state.scanned_image = image
                                st.session_state.characters = []
                                st.session_state.detections = []
                                
                                # Show result even if no face detected
                                final_display = image.copy()
                                final_display.thumbnail((400, 300), Image.Resampling.LANCZOS)
                                camera_placeholder.image(final_display, caption="No faces detected", use_container_width=True)
                                
                                st.warning("No faces detected! Try better lighting or positioning.")
                finally:
                    cap.release()
            
            st.rerun()
        
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.scanned_image = None
            st.session_state.characters = []
            st.session_state.detections = []
            # Reset camera display
            placeholder_img = load_placeholder_image()
            camera_placeholder.image(placeholder_img, caption="Ready to scan", use_container_width=True)
            st.rerun()
        
        # Quick stats
        if st.session_state.detections:
            st.metric("Faces Detected", len(st.session_state.detections))
            avg_confidence = sum(d['confidence'] for d in st.session_state.detections) / len(st.session_state.detections)
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Right column - Results
    with col3:
        st.subheader("✨ Your Destiny")
        
        display_character_info(st.session_state.characters)
        
        if st.session_state.characters:
            # Powers section
            st.markdown("**🔥 Your Powers:**")
            powers = [
                "Enhanced strength",
                "Combat expertise", 
                "Heroic determination",
                "Special abilities"
            ]
            for power in powers:
                st.write(f"• {power}")
            
            if len(st.session_state.characters) > 1:
                st.markdown("**🤝 Team Bonus:**")
                st.write("• Combined strength")
                st.write("• Coordinated attacks")
                st.write("• Shared wisdom")
    
    # Instructions at bottom (compact)
    with st.expander("📋 How to Use"):
        st.write("""
        **Quick Guide:**
        1. Click **'Scan Face'** - 3-second countdown, then records for 2 seconds
        2. **Position yourself** - Use the countdown to get ready and center your face(s)
        3. **Stay still** - Keep faces visible during the 2-second recording  
        4. **View results** - See superhero identity for each detected face
        5. **Scan again** - Click the button for a new scan
        
        💡 **Tips:** 
        - Good lighting and staying still during recording improve detection accuracy
        - Multiple faces will each get their own superhero identity
        - Face boxes are now better centered on actual faces
        """)

if __name__ == "__main__":
    main()
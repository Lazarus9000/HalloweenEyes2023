import cv2 
import numpy as np
import time
import pygame
import math

#https://simplegametutorials.github.io/pygamezero/eyes/

# pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
running = True

xbuff = [0] * 30
ybuff = [0] * 30

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Initialize the background subtractor (MOG2 method)
bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()

# Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 500
params.maxArea = 100000

params.filterByColor = True
params.blobColor = 255

params.filterByInertia = False;
params.filterByConvexity = False; 
# Create a blob detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Initialize webcam
cap = cv2.VideoCapture(0)

def draw_eye(eye_x, eye_y, lookx, looky):
    #mouse_x, mouse_y = pygame.mouse.get_pos()

    distance_x = lookx - eye_x
    distance_y = looky - eye_y

    distance = min(math.sqrt(distance_x**2 + distance_y**2), 90)
    angle = math.atan2(distance_y, distance_x)

    pupil_x = eye_x + (math.cos(angle) * distance)
    pupil_y = eye_y + (math.sin(angle) * distance)
    white = pygame.Color(255, 255, 255)
    blue = pygame.Color(0, 0, 100)
    eyecoord = (eye_x, eye_y)
    pupilcoord = (pupil_x, pupil_y)
    
    #https://www.pygame.org/docs/ref/draw.html#pygame.draw.circle
    pygame.draw.circle(screen, white, eyecoord, 150, 0) 
    pygame.draw.circle(screen, blue, pupilcoord, 50, 0)
    #screen.draw.filled_circle((eye_x, eye_y), 50, color=(255, 255, 255))
    #screen.draw.filled_circle((pupil_x, pupil_y), 15, color=(0, 0, 100))

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # fill the screen with a color to wipe away anything from last frame
    screen.fill("Black")
    
    # Apply background subtraction to the frame
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Perform morphological operations to clean up the mask
    #g_mask = cv2.erode(fg_mask, None, iterations=5)
    #fg_mask = cv2.dilate(fg_mask, None, iterations=10)

    # Detect blobs in the masked frame
    keypoints = detector.detect(fg_mask)

    # Find the largest blob
    largest_blob = None
    max_size = 0
    for keypoint in keypoints:
        if keypoint.size > max_size:
            max_size = keypoint.size
            largest_blob = keypoint
            print(keypoint.size)

    # Draw only the largest blob on the original frame
    if largest_blob is not None:
        print("Largest blob")
        print(largest_blob.size)
        fg_frame_with_blob = cv2.drawKeypoints(fg_mask, [largest_blob], 0, (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        frame_with_blob = cv2.drawKeypoints(frame, [largest_blob], 0, (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        
        h, w, c = frame.shape
        
        normalizedx = largest_blob.pt[0]*800/w
        
        #exxaggered x - to make eye movement more dramatic
        #width of eye picture 800. If it is exxagereted to 1400, then withdraw (1400-800)/2 = 300 to center
        scale = 1400
        exxagx = largest_blob.pt[0]/w*scale-((scale-800)/2)
        
        #Add values from found blob to circular buffer
        xbuff.pop(0)
        xbuff.append(exxagx)
        ybuff.pop(0)
        ybuff.append(700)
        #ybuff.append(largest_blob.pt[1]*600/480)
        
        
    else:
        frame_with_blob = frame
        fg_frame_with_blob = frame
        
        #No blob found, add default values to circular buffer
        xbuff.pop(0)
        xbuff.append(250)
        
        ybuff.pop(0)
        ybuff.append(200)

    # Display the frame with the largest blob
    cv2.imshow('Largest Blob Detection with Background Subtraction', frame_with_blob)
    # Show the foreground mask in a separate window
    cv2.imshow('Foreground Mask', fg_frame_with_blob)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #Calculate mean value of circular buffer to calculate a moving average - smooths eye motion
    xpos = sum(xbuff) / len(xbuff)
    ypos = sum(ybuff) / len(ybuff)
    
    draw_eye(200, 170, xpos, ypos)
    draw_eye(600, 170, xpos, ypos)
    
    #Draw eye display!
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

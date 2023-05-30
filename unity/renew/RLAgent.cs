using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using System;


public class RLAgent : MonoBehaviour
{
    Car_Controller carController;
    SocketClient socketClient;
    CollisionStateManager collisionStateManager;
    // Define the game state variables
    
    public Transform rewardCoinTransform;
    public Transform agentTransform;

    private byte[] gameState;

    public string FileName;
    public RenderTexture RT;
    public GameObject RenderCamera;
    public int reward;

    private float startTime;
    private float timeLimit;
    
    private float previousDistance;

    void Start()
    {
        collisionStateManager = GetComponent<CollisionStateManager>();
        socketClient = GetComponent<SocketClient>();
        agentTransform = transform;
        carController = GetComponent<Car_Controller>();
        
        initVar();
        InvokeRepeating("train", 0f, 0.3f);
    }

    void initVar() {
        previousDistance = getDistanceCoin();
        startTime = Time.time;
        timeLimit = 40f;
    }

    

    void train() {
        byte[] Imagebytes = getImage();


        byte isDone = collisionStateManager.isDone ? (byte)1 : (byte)0;
        float reward = getReward();
        float[] currentPosition = getCurrentPosition();
        
        socketClient.SendStates(isDone, reward, currentPosition, Imagebytes);
        string action = socketClient.GetAction();
        carController.Current_Virtual_Key = action;
        
        if (isDone == 1) endEpisode();
    }

    void endEpisode() {
        collisionStateManager.endEpisode();
        initVar();
    }

    float getDistanceCoin() {
        float distance = Vector3.Distance(rewardCoinTransform.position, agentTransform.position);
        return distance;
    }

    float getReward() {
        // float currentDistance = getDistanceCoin();


        float reward = -0.1f;
        // check in or out

        if (collisionStateManager.collisionType == 1) 
            reward += 10f;
        
        else if (collisionStateManager.collisionType == -1)
            reward -= 10f;

        // previousDistcance = currentDistance;
        collisionStateManager.collisionType = 0;

        return reward;
    }

    float[] getCurrentPosition() {
        float[] currentPosition = { 0f, 0f };
        currentPosition[0] = transform.position.x;
        currentPosition[1] = transform.position.z;
        return currentPosition;
    }

    public byte[] getImage() 
    {

        // while!!
        Texture2D texture2D = new Texture2D(RT.width, RT.height, TextureFormat.ARGB32, false);
        RenderTexture.active = RT;
        texture2D.ReadPixels(new Rect(0, 0, RT.width, RT.height), 0, 0);
        texture2D.Apply();

        // Resize the image to 84x84
        int targetWidth = 84;
        int targetHeight = 84;
        RenderTexture resizedRT = new RenderTexture(targetWidth, targetHeight, 24);
        Graphics.Blit(texture2D, resizedRT);

        // Create a new Texture2D with the resized dimensions
        Texture2D resizedTexture = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);
        RenderTexture.active = resizedRT;
        resizedTexture.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        resizedTexture.Apply();

        // Encode the resized texture to PNG
        byte[] imageBytes = resizedTexture.EncodeToPNG();

        string path = Application.persistentDataPath + "/" + FileName + ".png";

        if (imageBytes.Length > 90*90 + 50000)
        Debug.Log(imageBytes.Length);

        // Debug.Log(path);
        // File.WriteAllBytes(path, imageBytes);

        return imageBytes;
    }
    
    
}
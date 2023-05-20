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
        
        while (!socketClient.isConnected())
             Debug.Log("waiting");
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


        byte isDone = collisionStateManager.collisionType == 0 ? (byte)0 : (byte)1;
        float reward = getReward();
        float[] currentPosition = getCurrentPosition();
        

        float currentTime = Time.time;
        float timeDiff = currentTime - startTime;
        if (timeDiff > timeLimit) {
            isDone = (byte)1;
            socketClient.SendStates(isDone, reward, currentPosition, Imagebytes);
        }
        else {
            socketClient.SendStates(isDone, reward, currentPosition, Imagebytes);
            string action = socketClient.GetAction();
            carController.Current_Virtual_Key = action;
        }
        
        
        
        if (isDone == 1)
            endEpisode();
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
        float currentDistance = getDistanceCoin();

        float reward = 0;
        // check in or out
        
        if (currentDistance < previousDistance)
            reward = 0.1f; // Small reward for getting closer to a coin
        
        if (collisionStateManager.collisionType == 1) 
            reward += 1f;
        
        else if (collisionStateManager.collisionType == -1)
            reward -= 1f;

        previousDistance = currentDistance;

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

        string Path = Application.persistentDataPath + "/" + FileName + ".png";
        byte[] Imagebytes = texture2D.EncodeToPNG();

        if (Imagebytes.Length > 257*257 + 50000)
        Debug.Log(Imagebytes.Length);

        // File.WriteAllBytes(Path, Imagebytes);

        return Imagebytes;
    }
    
    
}
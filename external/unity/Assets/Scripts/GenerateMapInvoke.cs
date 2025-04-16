
using UnityEngine;
using System;
using System.Net;
using RESTfulHTTPServer.src.models;
using RESTfulHTTPServer.src.controller;


namespace RESTfulHTTPServer.src.invoker
{
    public class CoroutineExecuter : MonoBehaviour { }
    public class GenerateMapInvoke : MonoBehaviour
    {
        private const string TAG = "Gemerate Map Invoke";
        private static CoroutineExecuter instance;

        public static Response Generate(Request request)
        {
            Response response = new Response();
            string responseData = "";
            string json = request.GetPOSTData();
            bool valid = true;
            bool timedOut = false;

            UnityInvoker.ExecuteOnMainThread.Enqueue(() =>
            {

                try
                {
                    MapInvokerRequest jsonObj = MapInvokerRequest.CreateFromJSON(json);
                    string matrixString = jsonObj.matrixString;
                    ConvertMatrixToMap generator = GameObject.Find("Grid/FactoryMap").GetComponent<ConvertMatrixToMap>();
                    generator.Generate(matrixString);
                    if (!instance)
                    {
                        instance = FindObjectOfType<CoroutineExecuter>();
                        if (!instance)
                        {
                            instance = new GameObject("CoroutineExecuter").AddComponent<CoroutineExecuter>();
                        }
                    }
                    instance.StartCoroutine(generator.WaitTillSimulationComplete((isDone) =>
                    {
                        if (isDone)
                        {
                            if (generator.failed)
                            {
                                responseData = "Timed Out";
                                timedOut = true;
                                return;
                            }

                            GoalReached goalReached = generator.goal.GetComponent<GoalReached>();
                            SimulationResponse simulationResult = new SimulationResponse(goalReached.pathExists, goalReached.timedOut, goalReached.simulationTimer, goalReached.collisionCount, goalReached.proximityTimer);
                            response.SetHTTPStatusCode((int)HttpStatusCode.OK);
                            responseData = JsonUtility.ToJson(simulationResult);
                        }
                    }));

                }
                catch (Exception ex)
                {
                    valid = false;
                    string msg = "failed to deseiralised JSON";
                    responseData = msg;
                }

            });

            // Wait for the main thread
            while (responseData.Equals("")) ;
            print(responseData);

            // Filling up the response with data
            if (valid)
            {

                // 200 - OK
                response.SetContent(responseData);
                int statusCode = (timedOut)?(int)HttpStatusCode.RequestTimeout:(int)HttpStatusCode.OK;
                response.SetHTTPStatusCode(statusCode);
                response.SetMimeType(Response.MIME_CONTENT_TYPE_JSON);
            }
            else
            {

                // 406 - Not acceptable
                response.SetContent("Somthing wrong");
                response.SetHTTPStatusCode((int)HttpStatusCode.NotAcceptable);
                response.SetMimeType(Response.MIME_CONTENT_TYPE_HTML);
            }
            return response;
        }
    }
}
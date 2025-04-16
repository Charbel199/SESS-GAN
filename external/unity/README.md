# SimulatorGAN
Unity Repository for Simulation Environment Generation using GANs.

# Running headless mode
To run headless mode:

`./Headless/simgan.x86_64 -batchmode -nographics`

# API Request

* Currently, there is only a POST request possible.

* Post to **http:localhost:8080/simulate**

* Post a request with the following body:

    `{"matrixString": "1 0 0 0 0 1 1 1 0 0 0 1 0 1 \n 0 1 1 1...."}`

_Preferably, use **32x32** matrices (for now)_

* Response will contain the following:
    * Status: 
        * 200(OK): indicating success
        * 408 (REQUEST_TIMEOUT): indicating that the generation timed out
        * 406 (NOT_ACCEPTABLE): something unexpected 
    * Content:

        * 200:

            {
                "pathExists": true(bool), //this indicates if there is a path from agent to goal

                "timedOut": false(bool), //this indicates that the **simulation** timed out, meaning that the agent is probably stuck

                "simulationTime": 19.21 (float), //this indicates how long the simulation was

                "collisionCount": 4 (int), //this indicates how many times the agent collided with obstacles

                "proximityTime": 10.61 (float), //this indicates how much time the agent was close to an obstacle
            }
        
        * 408:
            "Timed Out"
        
        * 406:
            "Somthing wrong"

import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import numpy as np
def botResponse(prompt):
    loaded_model = pickle.load(open("finalized_model.pkl", 'rb'))
    x = np.array([64, 1, 0, 120, 246, 0, 0, 96, 1, 2.2, 0, 1, 2])
    x = x.reshape(1, -1)
    result = loaded_model.predict(x)
    return result

@app.get("/chat")
async def chat(query: str):
    return botResponse(query)


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, port=int(os.environ.get('PORT', 8081)), host="127.0.0.1")



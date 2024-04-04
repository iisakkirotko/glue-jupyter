import * as modelViewer from "https://unpkg.com/@google/model-viewer@3.3.0/dist/model-viewer.min.js"
import * as React from "react"

export default ( {model, viewer_height} ) => {
    const forceLoadModelViewer = modelViewer;
    const modelBlob = new Blob([model], {type: "application/octet-stream"});
    const url = URL.createObjectURL(modelBlob);
    return <model-viewer alt="Neil Armstrong's Spacesuit from the Smithsonian Digitization Programs Office and National Air and Space Museum"
    src={url}
    shadow-intensity="1"
    camera-controls 
    touch-action="pan-y"
    ar
    style={{height: viewer_height, width: "100%"}}></model-viewer> 
}

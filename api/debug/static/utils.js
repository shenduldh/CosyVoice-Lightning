async function base64ToAudioBuffer(base64String) {
    if (base64String.includes(",")) base64String = base64String.split(",")[1]
    binaryString = atob(base64String)
    const bytes = Uint8Array.from(binaryString, c => c.charCodeAt(0))
    const audioBuffer = await (new AudioContext()).decodeAudioData(bytes.buffer)
    return audioBuffer
}

async function arrayBufferToAudioBuffer(arrayBuffer) {
    const audioBuffer = await (new AudioContext()).decodeAudioData(arrayBuffer)
    return audioBuffer
}

function intToFloat(intArray) {
    const maxValue = Math.max(...intArray)
    const float32Array = new Float32Array(intArray.length)
    for (let i = 0; i < intArray.length; i++)
        float32Array[i] = intArray[i] / maxValue
    return float32Array
}

function floatToInt(float32Array, bitsPerSample) {
    const maxValue = Math.pow(2, bitsPerSample - 1)
    let IntArrayConstructor
    if (bitsPerSample === 32) IntArrayConstructor = Int32Array
    else if (bitsPerSample === 16) IntArrayConstructor = Int16Array
    else IntArrayConstructor = Int8Array

    const outputArray = new IntArrayConstructor(float32Array.length)
    for (let i = 0; i < float32Array.length; i++) {
        const sample = Math.max(-1, Math.min(1, float32Array[i]))
        outputArray[i] = sample < 0 ? sample * maxValue : sample * (maxValue - 1)
    }
    return outputArray
}

function setWavHeader(view, sampleRate, bitsPerSample, numberOfChannels, numberOfSamples) {
    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++)
            view.setUint8(offset + i, string.charCodeAt(i))
    }

    const bytesPerSample = bitsPerSample / 8
    const blockAlign = numberOfChannels * bytesPerSample
    const byteRate = sampleRate * blockAlign
    const dataSize = numberOfSamples * numberOfChannels * bytesPerSample

    writeString(view, 0, "RIFF")
    view.setUint32(4, 36 + dataSize, true)
    writeString(view, 8, "WAVE")
    writeString(view, 12, "fmt ")
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true)
    view.setUint16(22, numberOfChannels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, byteRate, true)
    view.setUint16(32, blockAlign, true)
    view.setUint16(34, bitsPerSample, true)
    writeString(view, 36, "data")
    view.setUint32(40, dataSize, true)
}

function int16ArrayBufferToWav(arrayBuffer, sampleRate, bitsPerSample, numberOfChannels) {
    let data = new Int16Array(arrayBuffer)
    if (bitsPerSample !== 16)
        data = floatToInt(intToFloat(data), bitsPerSample)

    const numberOfSamples = data.length
    const bytesPerSample = bitsPerSample / 8
    const dataSize = numberOfSamples * numberOfChannels * bytesPerSample
    const bufferSize = 44 + dataSize

    const outputBuffer = new ArrayBuffer(bufferSize)
    const view = new DataView(outputBuffer)

    setWavHeader(view, sampleRate, bitsPerSample, numberOfChannels, numberOfSamples)

    for (let i = 0, offset = 44; i < numberOfSamples; i++, offset += bytesPerSample)
        if (bitsPerSample === 32) view.setInt32(offset, data[i], true)
        else if (bitsPerSample === 16) view.setInt16(offset, data[i], true)
        else view.setInt8(offset, data[i], true)

    return outputBuffer
}

function audioBufferToWav(audioBuffer, bitsPerSample) {
    const numberOfChannels = audioBuffer.numberOfChannels
    const numberOfSamples = audioBuffer.length
    const sampleRate = audioBuffer.sampleRate
    const bytesPerSample = bitsPerSample / 8
    const dataSize = numberOfSamples * numberOfChannels * bytesPerSample
    const bufferSize = 44 + dataSize

    const outputBuffer = new ArrayBuffer(bufferSize)
    const view = new DataView(outputBuffer)

    setWavHeader(view, sampleRate, bitsPerSample, numberOfChannels, numberOfSamples)

    const data = []
    for (let i = 0; i < numberOfChannels; i++)
        data.push(floatToInt(audioBuffer.getChannelData(i), bitsPerSample))

    let offset = 44
    for (let i = 0; i < numberOfSamples; i++) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
            if (bitsPerSample === 32) view.setInt32(offset, data[channel][i], true)
            else if (bitsPerSample === 16) view.setInt16(offset, data[channel][i], true)
            else view.setInt8(offset, data[channel][i], true)
            offset += bytesPerSample
        }
    }

    return outputBuffer
}

function arrayBufferToBase64(arrayBuffer) {
    const bytes = new Uint8Array(arrayBuffer)
    let binaryString = ""
    for (let i = 0; i < bytes.byteLength; i++)
        binaryString += String.fromCharCode(bytes[i])
    return btoa(binaryString)
}
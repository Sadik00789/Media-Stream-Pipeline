package ipc

const (
	FrameSamples    = 512
	SpectrogramBins = 256
	RingCapacity    = 64
	ShmMagic        = 0x4D535452
	ShmVersion      = 1

	FlagIngested  = 1 << 0
	FlagDspDone   = 1 << 1
	FlagMlDone    = 1 << 2
	FlagBroadcast = 1 << 3

	VideoWidth        = 640
	VideoHeight       = 480
	VideoChannels     = 3
	VideoFrameSize    = VideoWidth * VideoHeight * VideoChannels // 921600 bytes
	VideoRingCapacity = 8
	VideoShmMagic     = 0x56494445 // 'VIDE'
	VideoShmVersion   = 1

	VideoFormatRGB24   = 0
	VideoFormatYUV420P = 1
	MaxDetections      = 16
)

type AudioFramePacket struct {
	SequenceNumber uint64
	TimestampNs    uint64
	SampleRate     uint32
	Channels       uint32
	SampleCount    uint32
	VadActive      uint32
	VadConfidence  float32
	MlConfidence   float32
	Flags          uint32
	Reserved       uint32
	PcmData        [FrameSamples]float32
	Spectrogram    [SpectrogramBins]float32
	MlTranscript   [128]byte
}

type SharedMemoryRingBuffer struct {
	Magic          uint32
	Version        uint32
	BufferCapacity uint32
	PacketSize     uint32
	WriteHead      uint64
	ReadHead       uint64
	DroppedFrames  uint64
	OverrunCount   uint64
	Packets        [RingCapacity]AudioFramePacket
}

type DetectionBox struct {
	Ymin       float32 `json:"ymin"`
	Xmin       float32 `json:"xmin"`
	Ymax       float32 `json:"ymax"`
	Xmax       float32 `json:"xmax"`
	Confidence float32 `json:"confidence"`
	ClassID    uint32  `json:"class_id"`
}

type VideoFramePacket struct {
	SequenceNumber uint64
	TimestampNs    uint64
	Width          uint32
	Height         uint32
	Format         uint32
	DetectionCount uint32
	Flags          uint32
	Reserved       uint32
	Detections     [MaxDetections]DetectionBox
	PixelData      [VideoFrameSize]byte
}

type SharedMemoryVideoRingBuffer struct {
	Magic          uint32
	Version        uint32
	BufferCapacity uint32
	PacketSize     uint32
	WriteHead      uint64
	ReadHead       uint64
	DroppedFrames  uint64
	OverrunCount   uint64
	Packets        [VideoRingCapacity]VideoFramePacket
}

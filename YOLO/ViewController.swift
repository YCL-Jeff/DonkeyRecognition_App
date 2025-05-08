import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision
import CoreImage

// YOLO 模型
var mlModel: MLModel = {
    guard let path = Bundle.main.path(forResource: "best", ofType: "mlmodelc") else {
        // 列出 Bundle 內容以調試
        let fileManager = FileManager.default
        if let bundlePath = Bundle.main.bundlePath as NSString? {
            do {
                let contents = try fileManager.contentsOfDirectory(atPath: bundlePath as String)
                print("Bundle 內容：\(contents)")
                let modelsPath = bundlePath.appendingPathComponent("Models")
                if fileManager.fileExists(atPath: modelsPath) {
                    let modelsContents = try fileManager.contentsOfDirectory(atPath: modelsPath)
                    print("Models/ 資料夾內容：\(modelsContents)")
                }
            } catch {
                print("無法列出 Bundle 內容：\(error)")
            }
        }
        fatalError("無法找到 best.mlmodelc 檔案，請確認檔案已正確添加到專案")
    }
    let modelURL = URL(fileURLWithPath: path)
    do {
        return try MLModel(contentsOf: modelURL, configuration: mlmodelConfig)
    } catch {
        fatalError("無法載入 best.mlmodelc 模型：\(error.localizedDescription)")
    }
}()

var mlmodelConfig: MLModelConfiguration = {
    let config = MLModelConfiguration()
    if #available(iOS 17.0, *) {
        config.setValue(1, forKey: "experimentalMLE5EngineUsage")
    }
    return config
}()

// MegaDescriptor 模型
var megaDescriptorModel: MLModel?

struct DetectionResult {
    let boundingBox: CGRect
    let name: String
    let confidence: Float
}

class ViewController: UIViewController {
    @IBOutlet var videoPreview: UIView!
    @IBOutlet var View0: UIView!
    @IBOutlet var playButtonOutlet: UIBarButtonItem!
    @IBOutlet var pauseButtonOutlet: UIBarButtonItem!
    @IBOutlet weak var labelName: UILabel!
    @IBOutlet weak var labelFPS: UILabel!
    @IBOutlet weak var labelZoom: UILabel!
    @IBOutlet weak var labelVersion: UILabel!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var focus: UIImageView!
    @IBOutlet weak var toolBar: UIToolbar!

    let selection = UISelectionFeedbackGenerator()
    var detector = try! VNCoreMLModel(for: mlModel)
    var session: AVCaptureSession!
    var videoCapture: VideoCapture!
    var currentBuffer: CVPixelBuffer?
    var framesDone = 0
    var t0 = 0.0
    var t1 = 0.0
    var t2 = 0.0
    var t3 = CACurrentMediaTime()
    var t4 = 0.0
    var longSide: CGFloat = 3
    var shortSide: CGFloat = 4
    var frameSizeCaptured = false
    var features: [[Float]] = []
    var labels: [String] = []
    var featureVector: [Float]?

    let donkeyClassLabel = "donkey"

    lazy var visionRequest: VNCoreMLRequest = {
        let request = VNCoreMLRequest(
            model: detector,
            completionHandler: { [weak self] request, error in
                self?.processObservations(for: request, error: error)
            })
        request.imageCropAndScaleOption = .scaleFill
        return request
    }()

    override func viewDidLoad() {
        super.viewDidLoad()
        // 設置 UserDefaults 中的 app_version（如果未設置）
        if UserDefaults.standard.string(forKey: "app_version") == nil {
            UserDefaults.standard.set("1.0.0", forKey: "app_version")
        }
        setLabels()
        setUpBoundingBoxViews()
        setUpOrientationChangeNotification()
        startVideo()
        setModel()

        // 異步載入特徵庫
        DispatchQueue.global(qos: .userInitiated).async {
            if let url = Bundle.main.url(forResource: "features_and_labels", withExtension: "json") {
                print("找到 features_and_labels.json：\(url.path)")
                do {
                    let data = try Data(contentsOf: url)
                    let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
                    
                    // 嘗試解析 features
                    if let rawFeatures = json?["features"] as? [[Any]] {
                        // 將 [[Any]] 轉換為 [[Float]]
                        self.features = rawFeatures.map { feature in
                            feature.compactMap { value in
                                if let floatValue = value as? Float {
                                    return floatValue
                                } else if let doubleValue = value as? Double {
                                    return Float(doubleValue)
                                } else if let intValue = value as? Int {
                                    return Float(intValue)
                                }
                                return nil
                            }
                        }
                    } else {
                        print("無法解析 features 數據，可能是格式不正確")
                        self.features = []
                    }

                    // 解析 labels
                    self.labels = (json?["labels"] as? [String]) ?? []
                    
                    print("成功載入 features_and_labels.json，features 數量：\(self.features.count)，labels 數量：\(self.labels.count)")
                    
                    // 檢查 features 和 labels 數量是否一致
                    if self.features.count != self.labels.count {
                        print("警告：features 和 labels 數量不一致，features: \(self.features.count), labels: \(self.labels.count)")
                    }
                    // 檢查第一個特徵向量的長度（應為 512）
                    if let firstFeature = self.features.first {
                        print("第一個特徵向量長度：\(firstFeature.count)")
                        if firstFeature.count != 512 {
                            print("警告：特徵向量長度不為 512，可能導致匹配失敗")
                        }
                    }
                } catch {
                    print("無法載入 features_and_labels.json 檔案，錯誤：\(error.localizedDescription)")
                }
            } else {
                print("無法找到 features_and_labels.json 檔案，請確認檔案已正確添加到專案")
            }
        }

        // 載入 MegaDescriptor 模型
        do {
            if let path = Bundle.main.path(forResource: "MegaDescriptor", ofType: "mlmodelc") {
                let modelURL = URL(fileURLWithPath: path)
                megaDescriptorModel = try MLModel(contentsOf: modelURL, configuration: MLModelConfiguration())
                print("成功載入 MegaDescriptor 模型")
            } else {
                print("無法找到 MegaDescriptor.mlmodelc 檔案，請確認檔案已正確添加到專案")
            }
        } catch {
            print("無法載入 MegaDescriptor 模型：\(error.localizedDescription)")
        }
    }

    override func viewWillTransition(to size: CGSize, with coordinator: any UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
        self.videoCapture.previewLayer?.frame = CGRect(x: 0, y: 0, width: size.width, height: size.height)
    }

    private func setUpOrientationChangeNotification() {
        NotificationCenter.default.addObserver(
            self, selector: #selector(orientationDidChange),
            name: UIDevice.orientationDidChangeNotification, object: nil)
    }

    @objc func orientationDidChange() {
        videoCapture.updateVideoOrientation()
    }

    @IBAction func vibrate(_ sender: Any) {
        selection.selectionChanged()
    }

    func setModel() {
        do {
            detector = try VNCoreMLModel(for: mlModel)
            detector.featureProvider = ThresholdProvider()
            let request = VNCoreMLRequest(
                model: detector,
                completionHandler: { [weak self] request, error in
                    self?.processObservations(for: request, error: error)
                })
            request.imageCropAndScaleOption = .scaleFill
            visionRequest = request
            t2 = 0.0
            t3 = CACurrentMediaTime()
            t4 = 0.0
        } catch {
            print("無法初始化 VNCoreMLModel：\(error.localizedDescription)")
        }
    }

    @IBAction func takePhoto(_ sender: Any?) {
        let settings = AVCapturePhotoSettings()
        usleep(20_000)
        self.videoCapture.cameraOutput.capturePhoto(with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    }

    @IBAction func logoButton(_ sender: Any) {
        selection.selectionChanged()
        if let link = URL(string: "https://www.ultralytics.com") {
            UIApplication.shared.open(link)
        }
    }

    func setLabels() {
        self.labelName?.text = "驢子辨識 (YOLO11)"
        self.labelVersion?.text = "Version " + (UserDefaults.standard.string(forKey: "app_version") ?? "Unknown")
    }

    @IBAction func playButton(_ sender: Any) {
        selection.selectionChanged()
        self.videoCapture.start()
        playButtonOutlet?.isEnabled = false
        pauseButtonOutlet?.isEnabled = true
    }

    @IBAction func pauseButton(_ sender: Any?) {
        selection.selectionChanged()
        self.videoCapture.stop()
        playButtonOutlet?.isEnabled = true
        pauseButtonOutlet?.isEnabled = false
    }

    @IBAction func shareButton(_ sender: Any) {
        selection.selectionChanged()
        let settings = AVCapturePhotoSettings()
        self.videoCapture.cameraOutput.capturePhoto(with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    }

    let maxBoundingBoxViews = 10
    var boundingBoxViews = [BoundingBoxView]()

    func setUpBoundingBoxViews() {
        while boundingBoxViews.count < maxBoundingBoxViews {
            boundingBoxViews.append(BoundingBoxView())
        }
    }

    func startVideo() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self

        videoCapture.setUp(sessionPreset: .hd1280x720) { success in
            if success {
                if let previewLayer = self.videoCapture.previewLayer {
                    self.videoPreview?.layer.addSublayer(previewLayer)
                    self.videoCapture.previewLayer?.frame = self.videoPreview?.bounds ?? CGRect.zero
                }

                for box in self.boundingBoxViews {
                    box.addToLayer(self.videoPreview?.layer ?? CALayer())
                }

                self.videoCapture.start()
            } else {
                print("無法初始化視訊捕捉")
            }
        }
    }

    func predict(sampleBuffer: CMSampleBuffer) {
        if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            currentBuffer = pixelBuffer
            if !frameSizeCaptured {
                let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                longSide = max(frameWidth, frameHeight)
                shortSide = min(frameWidth, frameHeight)
                frameSizeCaptured = true
            }

            let imageOrientation: CGImagePropertyOrientation
            switch UIDevice.current.orientation {
            case .portrait:
                imageOrientation = .up
            case .portraitUpsideDown:
                imageOrientation = .down
            case .landscapeLeft:
                imageOrientation = .up
            case .landscapeRight:
                imageOrientation = .up
            case .unknown:
                imageOrientation = .up
            default:
                imageOrientation = .up
            }

            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
            if UIDevice.current.orientation != .faceUp {
                t0 = CACurrentMediaTime()
                do {
                    try handler.perform([visionRequest])
                } catch {
                    print("視覺請求失敗：\(error.localizedDescription)")
                }
                t1 = CACurrentMediaTime() - t0
            }

            currentBuffer = nil
        }
    }

    func processObservations(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            if let results = request.results as? [VNRecognizedObjectObservation] {
                let donkeyResults = results.filter { observation in
                    observation.labels.contains { $0.identifier == self.donkeyClassLabel }
                }

                guard let pixelBuffer = self.currentBuffer else {
                    self.show(predictions: donkeyResults.map { DetectionResult(boundingBox: $0.boundingBox, name: "驢子", confidence: $0.confidence) })
                    return
                }
                let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

                var updatedResults: [DetectionResult] = []
                for observation in donkeyResults {
                    let imgWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                    let imgHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                    let bbox = self.yoloToPixel(bbox: observation.boundingBox, imgWidth: imgWidth, imgHeight: imgHeight)

                    guard !self.features.isEmpty, !self.labels.isEmpty,
                          let croppedBuffer = self.preprocessImage(ciImage, bbox: bbox),
                          let queryFeature = self.extractFeatures(from: croppedBuffer) else {
                        updatedResults.append(DetectionResult(boundingBox: observation.boundingBox, name: "驢子", confidence: observation.confidence))
                        print("無法進行特徵匹配，可能是 features 或 labels 為空，或特徵提取失敗")
                        continue
                    }

                    // 打印提取的特徵向量以調試
                    print("提取的特徵向量長度：\(queryFeature.count)")

                    var bestScore: Float = -1.0
                    var bestIndex = 0
                    for (index, feature) in self.features.enumerated() {
                        let score = self.cosineSimilarity(queryFeature, feature)
                        if score > bestScore {
                            bestScore = score
                            bestIndex = index
                        }
                    }

                    let donkeyName = self.labels[bestIndex].split(separator: "_").first ?? "驢子"
                    updatedResults.append(DetectionResult(boundingBox: observation.boundingBox, name: String(donkeyName), confidence: observation.confidence))
                    print("檢測到驢子：\(donkeyName)，置信度：\(observation.confidence * 100)%, 邊界框：\(observation.boundingBox)")
                }

                self.show(predictions: updatedResults)
            } else {
                self.show(predictions: [])
            }

            if self.t1 < 10.0 {
                self.t2 = self.t1 * 0.05 + self.t2 * 0.95
            }
            self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95
            self.labelFPS?.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)
            self.t3 = CACurrentMediaTime()
        }
    }

    func show(predictions: [DetectionResult]) {
        let width = videoPreview?.bounds.width ?? 0
        let height = videoPreview?.bounds.height ?? 0

        if UIDevice.current.orientation == .portrait {
            var ratio: CGFloat = 1.0
            if videoCapture.captureSession.sessionPreset == .photo {
                ratio = (height / width) / (4.0 / 3.0)
            } else {
                ratio = (height / width) / (16.0 / 9.0)
            }

            for i in 0..<boundingBoxViews.count {
                if i < predictions.count {
                    let prediction = predictions[i]

                    var rect = prediction.boundingBox
                    switch UIDevice.current.orientation {
                    case .portraitUpsideDown:
                        rect = CGRect(
                            x: 1.0 - rect.origin.x - rect.width,
                            y: 1.0 - rect.origin.y - rect.height,
                            width: rect.width,
                            height: rect.height)
                    case .landscapeLeft, .landscapeRight, .unknown:
                        break
                    default:
                        break
                    }

                    if ratio >= 1 {
                        let offset = (1 - ratio) * (0.5 - rect.minX)
                        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
                        rect = rect.applying(transform)
                        rect.size.width *= ratio
                    } else {
                        let offset = (ratio - 1) * (0.5 - rect.maxY)
                        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
                        rect = rect.applying(transform)
                        ratio = (height / width) / (3.0 / 4.0)
                        rect.size.height /= ratio
                    }

                    rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))

                    let label = String(format: "%@ %.1f%%", prediction.name, prediction.confidence * 100)
                    boundingBoxViews[i].show(frame: rect, label: label)
                } else {
                    boundingBoxViews[i].hide()
                }
            }
        } else {
            let frameAspectRatio = longSide / shortSide
            let viewAspectRatio = width / height
            var scaleX: CGFloat = 1.0
            var scaleY: CGFloat = 1.0
            var offsetX: CGFloat = 0.0
            var offsetY: CGFloat = 0.0

            if frameAspectRatio > viewAspectRatio {
                scaleY = height / shortSide
                scaleX = scaleY
                offsetX = (longSide * scaleX - width) / 2
            } else {
                scaleX = width / longSide
                scaleY = scaleX
                offsetY = (shortSide * scaleY - height) / 2
            }

            for i in 0..<boundingBoxViews.count {
                if i < predictions.count {
                    let prediction = predictions[i]

                    var rect = prediction.boundingBox
                    rect.origin.x = rect.origin.x * longSide * scaleX - offsetX
                    rect.origin.y = height - (rect.origin.y * shortSide * scaleY - offsetY + rect.size.height * shortSide * scaleY)
                    rect.size.width *= longSide * scaleX
                    rect.size.height *= shortSide * scaleY

                    let label = String(format: "%@ %.1f%%", prediction.name, prediction.confidence * 100)
                    boundingBoxViews[i].show(frame: rect, label: label)
                } else {
                    boundingBoxViews[i].hide()
                }
            }
        }
    }

    let minimumZoom: CGFloat = 1.0
    let maximumZoom: CGFloat = 1.0
    var lastZoomFactor: CGFloat = 1.0

    @IBAction func pinch(_ pinch: UIPinchGestureRecognizer) {
        let device = videoCapture.captureDevice

        func minMaxZoom(_ factor: CGFloat) -> CGFloat {
            return min(min(max(factor, minimumZoom), maximumZoom), device.activeFormat.videoMaxZoomFactor)
        }

        func update(scale factor: CGFloat) {
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }
                device.videoZoomFactor = factor
            } catch {
                print("\(error.localizedDescription)")
            }
        }

        let newScaleFactor = minMaxZoom(pinch.scale * lastZoomFactor)
        switch pinch.state {
        case .began, .changed:
            update(scale: newScaleFactor)
            self.labelZoom?.text = String(format: "%.2fx", newScaleFactor)
            self.labelZoom?.font = UIFont.preferredFont(forTextStyle: .title2)
        case .ended:
            lastZoomFactor = minMaxZoom(newScaleFactor)
            update(scale: lastZoomFactor)
            self.labelZoom?.font = UIFont.preferredFont(forTextStyle: .body)
        default:
            break
        }
    }

    func yoloToPixel(bbox: CGRect, imgWidth: CGFloat, imgHeight: CGFloat) -> CGRect {
        let xMin = bbox.origin.x * imgWidth
        let yMin = bbox.origin.y * imgHeight
        let width = bbox.width * imgWidth
        let height = bbox.height * imgHeight
        return CGRect(x: xMin, y: yMin, width: width, height: height)
    }

    func preprocessImage(_ image: CIImage, bbox: CGRect) -> CVPixelBuffer? {
        let croppedImage = image.cropped(to: bbox)

        let context = CIContext()
        let resizedImage = croppedImage.transformed(by: CGAffineTransform(scaleX: 224.0 / croppedImage.extent.width, y: 224.0 / croppedImage.extent.height))

        guard let cgImage = context.createCGImage(resizedImage, from: resizedImage.extent) else {
            return nil
        }

        let width = 224
        let height = 224
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let baseAddress = CVPixelBufferGetBaseAddress(buffer)!
        let context2 = CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        context2.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, [])

        return buffer
    }

    func extractFeatures(from pixelBuffer: CVPixelBuffer) -> [Float]? {
        guard let megaDescriptorModel = megaDescriptorModel else {
            print("MegaDescriptor 模型未載入")
            return nil
        }

        guard let megaDescriptor = try? VNCoreMLModel(for: megaDescriptorModel) else {
            print("無法創建 MegaDescriptor 的 VNCoreMLModel")
            return nil
        }

        let request = VNCoreMLRequest(model: megaDescriptor) { request, error in
            if let error = error {
                print("特徵提取失敗：\(error.localizedDescription)")
                return
            }
            guard let results = request.results as? [VNFeaturePrintObservation],
                  let feature = results.first?.data else {
                print("無法從 MegaDescriptor 提取特徵")
                return
            }
            self.featureVector = feature.map { Float($0) }
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("執行特徵提取請求失敗：\(error.localizedDescription)")
            return nil
        }

        if let featureVector = self.featureVector {
            print("成功提取特徵向量，長度：\(featureVector.count)")
            return featureVector
        } else {
            print("特徵向量為空")
            return nil
        }
    }

    func cosineSimilarity(_ vectorA: [Float], _ vectorB: [Float]) -> Float {
        guard vectorA.count == vectorB.count else {
            print("特徵向量長度不匹配：vectorA 長度 \(vectorA.count)，vectorB 長度 \(vectorB.count)")
            return 0.0
        }
        let dotProduct = zip(vectorA, vectorB).map(*).reduce(0, +)
        let magnitudeA = sqrt(vectorA.map { $0 * $0 }.reduce(0, +))
        let magnitudeB = sqrt(vectorB.map { $0 * $0 }.reduce(0, +))
        guard magnitudeA != 0, magnitudeB != 0 else {
            print("特徵向量幅度為 0：magnitudeA \(magnitudeA)，magnitudeB \(magnitudeB)")
            return 0.0
        }
        return dotProduct / (magnitudeA * magnitudeB)
    }
}

extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
        predict(sampleBuffer: sampleBuffer)
    }
}

extension ViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("error occurred : \(error.localizedDescription)")
            return
        }
        guard let dataImage = photo.fileDataRepresentation(),
              let dataProvider = CGDataProvider(data: dataImage as CFData),
              let cgImageRef = CGImage(
                  jpegDataProviderSource: dataProvider,
                  decode: nil,
                  shouldInterpolate: true,
                  intent: .defaultIntent
              ) else {
            print("無法創建 CGImage")
            return
        }

        var isCameraFront = false
        if let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput,
           currentInput.device.position == .front {
            isCameraFront = true
        }
        var orientation: CGImagePropertyOrientation = isCameraFront ? .leftMirrored : .right
        switch UIDevice.current.orientation {
        case .landscapeLeft:
            orientation = isCameraFront ? .downMirrored : .up
        case .landscapeRight:
            orientation = isCameraFront ? .upMirrored : .down
        default:
            break
        }
        var image = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: .right)
        if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
           let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent) {
            image = UIImage(cgImage: cgImage)
        }
        let imageView = UIImageView(image: image)
        imageView.contentMode = .scaleAspectFill
        imageView.frame = videoPreview?.frame ?? CGRect.zero
        let imageLayer = imageView.layer
        videoPreview?.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

        let bounds = UIScreen.main.bounds
        UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
        self.View0?.drawHierarchy(in: bounds, afterScreenUpdates: true)
        guard let img = UIGraphicsGetImageFromCurrentImageContext() else {
            print("無法生成分享圖片")
            UIGraphicsEndImageContext()
            return
        }
        UIGraphicsEndImageContext()
        imageLayer.removeFromSuperlayer()
        let activityViewController = UIActivityViewController(
            activityItems: [img], applicationActivities: nil)
        activityViewController.popoverPresentationController?.sourceView = self.View0
        self.present(activityViewController, animated: true, completion: nil)
    }
}

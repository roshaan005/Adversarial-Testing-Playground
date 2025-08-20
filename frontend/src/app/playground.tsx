"use client"

import React, { useState } from "react"
import { Upload, Play, Download, BarChart3, Shield, Target, Brain, Zap, AlertCircle } from "lucide-react"
import { api, AttackResponse } from "../lib/api"

// Custom Button Component
const Button = ({
  children,
  onClick,
  disabled = false,
  variant = "default",
  className = "",
  ...props
}: {
  children: React.ReactNode
  onClick?: () => void
  disabled?: boolean
  variant?: "default" | "outline"
  className?: string
}) => {
  const baseClasses = "px-4 py-2 rounded-md font-medium transition-all duration-300 flex items-center justify-center"
  const variantClasses =
    variant === "outline"
      ? "border border-pink-500 text-pink-400 hover:bg-pink-500/10 bg-black"
      : "bg-gradient-to-r from-pink-600 to-pink-700 hover:from-pink-700 hover:to-pink-800 text-white"
  const disabledClasses = disabled ? "opacity-50 cursor-not-allowed" : "hover:scale-105 hover:shadow-lg"

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variantClasses} ${disabledClasses} ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}

// Custom Card Components
const Card = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-gray-900 border border-gray-800 rounded-lg shadow-2xl ${className}`}>{children}</div>
)

const CardHeader = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <div className={`p-6 border-b border-gray-800 ${className}`}>{children}</div>
)

const CardTitle = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <h3 className={`text-lg font-semibold text-pink-300 ${className}`}>{children}</h3>
)

const CardContent = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <div className={`p-6 ${className}`}>{children}</div>
)

// Custom Input Component
const Input = ({ className = "", ...props }: React.InputHTMLAttributes<HTMLInputElement>) => (
  <input
    className={`w-full px-3 py-2 bg-black border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-pink-500 focus:border-pink-500 ${className}`}
    {...props}
  />
)

// Custom Label Component
const Label = ({ children, className = "", ...props }: React.LabelHTMLAttributes<HTMLLabelElement>) => (
  <label className={`block text-sm font-medium text-gray-300 mb-2 ${className}`} {...props}>
    {children}
  </label>
)

// Custom Slider Component
const Slider = ({
  value,
  onValueChange,
  min = 0,
  max = 100,
  step = 1,
  className = "",
}: {
  value: number[]
  onValueChange: (value: number[]) => void
  min?: number
  max?: number
  step?: number
  className?: string
}) => (
  <input
    type="range"
    min={min}
    max={max}
    step={step}
    value={value[0]}
    onChange={(e) => onValueChange([Number.parseFloat(e.target.value)])}
    className={`w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer slider ${className}`}
    style={{
      background: `linear-gradient(to right, #ec4899 0%, #ec4899 ${((value[0] - min) / (max - min)) * 100}%, #374151 ${((value[0] - min) / (max - min)) * 100}%, #374151 100%)`,
    }}
  />
)

// Custom Select Components
const Select = ({
  children,
  value,
  onValueChange,
}: {
  children: React.ReactNode
  value: string
  onValueChange: (value: string) => void
}) => {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-3 py-2 bg-black border border-gray-700 rounded-md text-white text-left focus:outline-none focus:ring-2 focus:ring-pink-500"
      >
        {value || "Select an option"}
      </button>
      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-gray-900 border border-gray-700 rounded-md shadow-lg">
          {React.Children.map(children, (child) => {
            const childElement = child as React.ReactElement<{ value: string; onClick?: () => void }>
            return React.cloneElement(childElement, {
              onClick: () => {
                onValueChange(childElement.props.value)
                setIsOpen(false)
              },
            })
          })}
        </div>
      )}
    </div>
  )
}

const SelectItem = ({
  children,
  value,
  onClick,
}: {
  children: React.ReactNode
  value: string
  onClick?: () => void
}) => (
  <div onClick={onClick} className="px-3 py-2 hover:bg-gray-800 cursor-pointer text-white">
    {children}
  </div>
)

// Custom Tabs Components
const Tabs = ({
  children,
  defaultValue,
}: {
  children: React.ReactNode
  defaultValue: string
}) => {
  const [activeTab, setActiveTab] = useState(defaultValue)

  return (
    <div className="w-full">
      {React.Children.map(children, (child) => {
        const childElement = child as React.ReactElement<{ activeTab?: string; setActiveTab?: (tab: string) => void }>
        return React.cloneElement(childElement, { activeTab, setActiveTab })
      })}
    </div>
  )
}

const TabsList = ({
  children,
  activeTab,
  setActiveTab,
  className = "",
}: {
  children: React.ReactNode
  activeTab?: string
  setActiveTab?: (tab: string) => void
  className?: string
}) => (
  <div className={`grid grid-cols-3 bg-gray-900 border border-gray-800 rounded-md ${className}`}>
    {React.Children.map(children, (child) => {
      const childElement = child as React.ReactElement<{ activeTab?: string; setActiveTab?: (tab: string) => void }>
      return React.cloneElement(childElement, { activeTab, setActiveTab })
    })}
  </div>
)

const TabsTrigger = ({
  children,
  value,
  activeTab,
  setActiveTab,
  className = "",
}: {
  children: React.ReactNode
  value: string
  activeTab?: string
  setActiveTab?: (tab: string) => void
  className?: string
}) => (
  <button
    onClick={() => setActiveTab?.(value)}
    className={`px-4 py-2 text-xs font-medium rounded-md transition-colors ${
      activeTab === value ? "bg-pink-600 text-white" : "text-gray-300 hover:text-white hover:bg-gray-800"
    } ${className}`}
  >
    {children}
  </button>
)

const TabsContent = ({
  children,
  value,
  activeTab,
  className = "",
}: {
  children: React.ReactNode
  value: string
  activeTab?: string
  className?: string
}) => {
  if (activeTab !== value) return null

  return <div className={`mt-4 ${className}`}>{children}</div>
}

// Custom Badge Component
const Badge = ({
  children,
  variant = "default",
  className = "",
}: {
  children: React.ReactNode
  variant?: "default" | "outline"
  className?: string
}) => {
  const variantClasses =
    variant === "outline"
      ? "border border-pink-600 text-pink-400 bg-transparent"
      : "bg-pink-900/30 text-pink-300 border border-pink-700"

  return <span className={`px-2 py-1 text-xs font-medium rounded-md ${variantClasses} ${className}`}>{children}</span>
}

// Custom Progress Component
const Progress = ({
  value,
  className = "",
}: {
  value: number
  className?: string
}) => (
  <div className={`w-full bg-gray-800 rounded-full h-2 ${className}`}>
    <div
      className="bg-gradient-to-r from-pink-500 to-pink-600 h-2 rounded-full transition-all duration-300"
      style={{ width: `${value}%` }}
    />
  </div>
)

export default function AdversarialAttackPlayground() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [epsilonValue, setEpsilonValue] = useState([0.03])
  const [iterationsValue, setIterationsValue] = useState([10])
  const [learningRateValue, setLearningRateValue] = useState([0.01])
  const [selectedModel, setSelectedModel] = useState("resnet18")
  const [selectedAttack, setSelectedAttack] = useState("fgsm")
  const [isAttackRunning, setIsAttackRunning] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [attackResults, setAttackResults] = useState<AttackResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setUploadedFile(file)
      const reader = new FileReader()
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)
      setError(null)
      setShowResults(false)
      setAttackResults(null)
    }
  }

  const handleRunAttack = async () => {
    if (!uploadedFile) {
      setError("Please upload an image first")
      return
    }

    setIsAttackRunning(true)
    setError(null)
    
    try {
      const results = await api.runAttack({
        image: uploadedFile,
        attack: selectedAttack,
        model: selectedModel,
        epsilon: epsilonValue[0],
        alpha: learningRateValue[0],
        iterations: iterationsValue[0],
      })
      
      setAttackResults(results)
      setShowResults(true)
    } catch (err) {
      console.error('Attack failed:', err)
      setError(err instanceof Error ? err.message : 'Failed to run attack. Please try again.')
    } finally {
      setIsAttackRunning(false)
    }
  }

  return (
    <div className="h-screen bg-black flex">
      {/* Main Content Area */}
      <div className="flex-1 h-full overflow-y-auto">
        <div className="p-6">
          <div className="max-w-4xl mx-auto">
            <div className="mb-6">
              <h1 className="text-3xl font-bold bg-gradient-to-r from-pink-400 to-pink-600 bg-clip-text text-transparent mb-2">
                Adversarial Attack Testing Playground
              </h1>
              <p className="text-gray-300">Upload an image and test various adversarial attacks against ML models</p>
            </div>

            {error && (
              <div className="mb-6 p-4 bg-red-900/30 border border-red-800 rounded-lg flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <div>
                  <h3 className="text-red-300 font-medium">Error</h3>
                  <p className="text-red-400 text-sm">{error}</p>
                </div>
              </div>
            )}

            {/* Image Upload Area */}
            <Card className="mb-6">
              <CardContent className="p-4">
                {!uploadedImage ? (
                  <div className="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center hover:border-pink-500 transition-colors bg-gradient-to-br from-gray-900 to-black">
                    <div className="space-y-3">
                      <Upload className="w-12 h-12 mx-auto text-gray-500" />
                      <div>
                        <h3 className="text-lg font-medium text-white mb-2">Upload an image</h3>
                        <p className="text-gray-400 mb-4">Drag and drop an image here, or click to select</p>
                        <Input type="file" accept="image/*" onChange={handleImageUpload} className="max-w-xs mx-auto" />
                      </div>
                    </div>
                  </div>
                ) : showResults ? (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="text-xl font-semibold bg-gradient-to-r from-pink-400 to-pink-600 bg-clip-text text-transparent">
                        Attack Results
                      </h3>
                      <Button
                        variant="outline"
                        onClick={() => {
                          setShowResults(false)
                          setUploadedImage(null)
                        }}
                      >
                        <Upload className="w-4 h-4 mr-2" />
                        New Test
                      </Button>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-3">
                        <h4 className="font-medium text-gray-200">Original Image</h4>
                        <div className="relative group">
                          <img
                            src={uploadedImage || "/placeholder.svg?height=300&width=300"}
                            alt="Original"
                            className="w-full rounded-lg border border-gray-700 shadow-md transition-transform duration-300 group-hover:scale-105"
                          />
                        </div>
                        <div className="bg-green-900/30 p-3 rounded-lg border border-green-800">
                          <p className="text-sm text-green-300">
                            <span className="font-medium">Prediction:</span> {attackResults?.original.label || 'N/A'} ({attackResults?.original.confidence || 0}%)
                          </p>
                          <p className="text-xs text-green-400 mt-1">Confidence: {attackResults?.original.confidence && attackResults.original.confidence > 80 ? 'High' : attackResults?.original.confidence && attackResults.original.confidence > 50 ? 'Medium' : 'Low'}</p>
                        </div>
                      </div>

                      <div className="space-y-3">
                        <h4 className="font-medium text-gray-200">Adversarial Image</h4>
                        <div className="relative group">
                          <img
                            src={uploadedImage || "/placeholder.svg?height=300&width=300"}
                            alt="Adversarial"
                            className="w-full rounded-lg border border-gray-700 shadow-md transition-transform duration-300 group-hover:scale-105 filter contrast-110 brightness-95"
                          />
                        </div>
                        <div className="bg-pink-900/30 p-3 rounded-lg border border-pink-800">
                          <p className="text-sm text-pink-300">
                            <span className="font-medium">Prediction:</span> {attackResults?.adversarial.label || 'N/A'} ({attackResults?.adversarial.confidence || 0}%)
                          </p>
                          <p className="text-xs text-pink-400 mt-1">
                            Attack Success: {attackResults?.original.label !== attackResults?.adversarial.label ? 'True' : 'False'}
                          </p>
                        </div>
                      </div>
                    </div>

                    {attackResults && (
                      <div className="mt-6 p-4 bg-gray-900/50 rounded-lg border border-gray-700">
                        <h4 className="font-medium text-gray-200 mb-3">Attack Details</h4>
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                          <div>
                            <span className="text-gray-400">Model:</span>
                            <p className="text-white font-medium">{attackResults.model?.toUpperCase() || 'N/A'}</p>
                          </div>
                          <div>
                            <span className="text-gray-400">Attack Type:</span>
                            <p className="text-white font-medium">{attackResults.attack.toUpperCase()}</p>
                          </div>
                          <div>
                            <span className="text-gray-400">Epsilon:</span>
                            <p className="text-white font-medium">{attackResults.epsilon}</p>
                          </div>
                          <div>
                            <span className="text-gray-400">Alpha:</span>
                            <p className="text-white font-medium">{attackResults.alpha}</p>
                          </div>
                          <div>
                            <span className="text-gray-400">Iterations:</span>
                            <p className="text-white font-medium">{attackResults.iterations}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center hover:border-pink-500 transition-colors bg-gradient-to-br from-gray-900 to-black">
                    {isAttackRunning ? (
                      <div className="space-y-4">
                        <div className="w-12 h-12 mx-auto">
                          <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-700 border-t-pink-500"></div>
                        </div>
                        <div>
                          <h3 className="text-lg font-medium text-white mb-2">Running Attack...</h3>
                          <p className="text-gray-400">Processing adversarial perturbations</p>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <img
                          src={uploadedImage || "/placeholder.svg?height=200&width=300"}
                          alt="Uploaded"
                          className="max-w-sm max-h-48 mx-auto rounded-lg shadow-md transition-transform duration-300 hover:scale-105"
                        />
                        <div className="flex gap-4 justify-center">
                          <Button variant="outline" onClick={() => setUploadedImage(null)}>
                            <Upload className="w-4 h-4 mr-2" />
                            Upload New Image
                          </Button>
                          <Button onClick={handleRunAttack} disabled={isAttackRunning}>
                            <Play className="w-4 h-4 mr-2" />
                            Run Attack
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Right Sidebar */}
      <div className="w-96 bg-black border-l border-gray-800 shadow-2xl h-full overflow-y-auto">
        <div className="p-6">
          <Tabs defaultValue="models">
            <TabsList>
              <TabsTrigger value="models">Models</TabsTrigger>
              <TabsTrigger value="attacks">Attacks</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
            </TabsList>

            {/* Models Section */}
            <TabsContent value="models" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    Target Models
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label>Select Model</Label>
                    <Select value={selectedModel} onValueChange={setSelectedModel}>
                      <SelectItem value="resnet18">ResNet-18</SelectItem>
                      <SelectItem value="mobilenet_v2">MobileNet V2</SelectItem>
                    </Select>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Attacks Section */}
            <TabsContent value="attacks" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="w-5 h-5" />
                    Attack Configuration
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label>Attack Method</Label>
                    <Select value={selectedAttack} onValueChange={setSelectedAttack}>
                      <SelectItem value="fgsm">FGSM (Fast Gradient Sign Method)</SelectItem>
                      <SelectItem value="pgd">PGD (Projected Gradient Descent)</SelectItem>
                    </Select>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <Label>Epsilon (ε): {epsilonValue[0]}</Label>
                      <Slider value={epsilonValue} onValueChange={setEpsilonValue} max={0.3} min={0.001} step={0.001} />
                    </div>

                    <div>
                      <Label>Iterations: {iterationsValue[0]}</Label>
                      <Slider value={iterationsValue} onValueChange={setIterationsValue} max={100} min={1} step={1} />
                    </div>

                    <div>
                      <Label>Learning Rate: {learningRateValue[0]}</Label>
                      <Slider
                        value={learningRateValue}
                        onValueChange={setLearningRateValue}
                        max={0.1}
                        min={0.001}
                        step={0.001}
                      />
                    </div>
                  </div>

                  <div className="pt-4 space-y-2">
                    <Button 
                      className="w-full" 
                      onClick={handleRunAttack}
                      disabled={isAttackRunning || !uploadedFile}
                    >
                      <Zap className="w-4 h-4 mr-2" />
                      {isAttackRunning ? 'Running Attack...' : 'Launch Attack'}
                    </Button>
                    <Button 
                      variant="outline" 
                      className="w-full bg-transparent"
                      disabled={!attackResults}
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Export Results
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Analytics Section */}
            <TabsContent value="analytics" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" />
                    Attack Analytics
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <h4 className="font-medium text-sm text-gray-300">Available Models</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-sm">ResNet-18</span>
                        <Badge variant="outline">Available</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-sm">MobileNet V2</span>
                        <Badge variant="outline">Available</Badge>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <h4 className="font-medium text-sm text-gray-300">Attack Methods</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-sm">FGSM</span>
                        <Badge variant="outline">Single-step</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-sm">PGD</span>
                        <Badge variant="outline">Multi-step</Badge>
                      </div>
                    </div>
                  </div>

                  <div className="pt-4">
                    <Button variant="outline" className="w-full bg-transparent">
                      <Shield className="w-4 h-4 mr-2" />
                      Generate Report
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      <style jsx global>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #ec4899;
          cursor: pointer;
          border: 2px solid #ec4899;
        }
        
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #ec4899;
          cursor: pointer;
          border: 2px solid #ec4899;
        }
      `}</style>
    </div>
  )
}

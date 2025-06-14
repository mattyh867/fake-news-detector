import './App.css'
import ResultCard from './components/ResultCard'

function App() {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4">
        <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">
          Fake News Detector
        </h1>
        <ResultCard />
      </div>
    </div>
  )
}

export default App

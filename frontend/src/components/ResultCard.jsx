const ResultCard = () => {
  return (
    <div className="w-full max-w-4xl mx-auto p-6">
      {/* Input Section */}
      <div className="mb-8">
        <label 
          htmlFor="article-text" 
          className="block text-lg font-semibold mb-2 text-gray-700"
        >
          Enter News Article Text
        </label>
        <textarea
          id="article-text"
          className="w-full h-48 p-4 border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
          placeholder="Paste your article text here..."
        />
      </div>

      {/* Results Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold mb-4 text-gray-800">Analysis Results</h3>
        
        {/* Article Preview */}
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-600 mb-2">Article Preview:</h4>
          <p className="text-gray-700 italic">
            "Lorem ipsum dolor sit amet consectetur... (first 100 characters)"
          </p>
        </div>

        {/* Prediction Result */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <span className="text-lg font-bold text-red-600">FAKE</span>
            <span className="text-sm text-gray-500 ml-2">
              (98% confidence)
            </span>
          </div>
          <div className="bg-red-100 text-red-800 text-sm font-semibold px-4 py-1 rounded-full">
            High Risk
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className="bg-red-600 h-2.5 rounded-full" 
            style={{ width: '98%' }}
          ></div>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;

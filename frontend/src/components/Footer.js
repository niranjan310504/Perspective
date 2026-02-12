import React from 'react';

/**
 * Footer Component
 */
function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200 mt-12">
      <div className="container mx-auto px-4 py-8">
        <div className="grid md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">About Perspective</h4>
            <p className="text-sm text-gray-600">
              Perspective is an Indian media bias detection tool that uses fine-tuned 
              BERT models to identify 7 types of bias in news articles. Built as a 
              final year engineering project.
            </p>
          </div>

          {/* Tech Stack */}
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">Tech Stack</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• BERT (bert-base-uncased)</li>
              <li>• PyTorch & HuggingFace Transformers</li>
              <li>• Flask REST API</li>
              <li>• React & Tailwind CSS</li>
            </ul>
          </div>

          {/* Links */}
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">Links</h4>
            <ul className="text-sm space-y-1">
              <li>
                <a 
                  href="https://github.com/perspective-media-bias/perspective" 
                  className="text-primary-600 hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  GitHub Repository
                </a>
              </li>
              <li>
                <a 
                  href="/api/docs" 
                  className="text-primary-600 hover:underline"
                >
                  API Documentation
                </a>
              </li>
              <li>
                <a 
                  href="#about" 
                  className="text-primary-600 hover:underline"
                >
                  Learn About Bias Types
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Copyright */}
        <div className="mt-8 pt-6 border-t border-gray-200 text-center text-sm text-gray-500">
          <p>© 2025-26 Perspective Team. Final Year Project.</p>
          <p className="mt-1">
            Made with ❤️ for transparent journalism in India
          </p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;

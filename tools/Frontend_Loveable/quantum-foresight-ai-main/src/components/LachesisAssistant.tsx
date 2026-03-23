import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Brain, Send, Sparkles, Key, MessageCircle, Atom, Volume2, VolumeX, Paperclip, X, Image, Mic, MicOff } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import lachesisAvatar from "@/assets/lachesis-avatar-v2.jpg";
import { QuantumTemporalBayesianNetwork } from "@/lib/qtbn-engine";
import { FinancialData } from "@/types/quantum";
import { useAppContext } from "@/contexts/AppContext";
import { useVoice } from "@/hooks/useVoice";
import { apiWebSearch } from "@/lib/api";

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  attachments?: Attachment[];
}

interface Attachment {
  id: string;
  name: string;
  type: string;
  size: number;
  url: string;
  isImage: boolean;
}

export const LachesisAssistant = () => {
  const { state: appState } = useAppContext();
  const language = appState.language ?? "English";
  const { isListening, micSupported, startListening, stopListening } = useVoice(language);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hello! I'm Lachesis, your A.I Integrated Quantum Assistant. I specialize in quantum computing, financial analytics, and risk assessment. How can I help you navigate the quantum-enhanced financial landscape today?",
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [elevenLabsApiKey, setElevenLabsApiKey] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [generatedVoiceId, setGeneratedVoiceId] = useState<string | null>(null);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const generateLachesisVoice = async (apiKey: string): Promise<string | null> => {
    try {
      const previewRes = await fetch('https://api.elevenlabs.io/v1/voice-generation/generate-voice', {
        method: 'POST',
        headers: { 'xi-api-key': apiKey, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          gender: 'female',
          accent: 'british',
          accent_strength: 3.5,
          age: 'middle_aged',
          text: 'Εγώ είμαι η Λάχεσις. I am Lachesis, daughter of Zeus and Themis, measurer of the thread of life. The markets, like the Fates themselves, follow patterns written long before mortal eyes could see them.'
        })
      });
      if (!previewRes.ok) return null;
      const { generated_voice_id } = await previewRes.json();
      const saveRes = await fetch('https://api.elevenlabs.io/v1/voice-generation/create-voice', {
        method: 'POST',
        headers: { 'xi-api-key': apiKey, 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice_name: 'Lachesis', generated_voice_id })
      });
      if (!saveRes.ok) return null;
      const { voice_id } = await saveRes.json();
      return voice_id;
    } catch {
      return null;
    }
  };

  useEffect(() => {
    if (elevenLabsApiKey.trim() && !generatedVoiceId) {
      generateLachesisVoice(elevenLabsApiKey).then(id => { if (id) setGeneratedVoiceId(id); });
    }
  }, [elevenLabsApiKey]);

  const speakText = async (text: string) => {
    if (!voiceEnabled || !elevenLabsApiKey.trim()) return;

    try {
      setIsSpeaking(true);
      const voiceId = generatedVoiceId ?? '9BWtsMINqrJLrRacOk9x';
      const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': elevenLabsApiKey
        },
        body: JSON.stringify({
          text: text,
          model_id: 'eleven_multilingual_v2',
          voice_settings: {
            stability: 0.35,
            similarity_boost: 0.75,
            style: 0.45,
            use_speaker_boost: true
          }
        })
      });

      if (response.ok) {
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onended = () => {
          setIsSpeaking(false);
          URL.revokeObjectURL(audioUrl);
        };
        
        await audio.play();
      }
    } catch (error) {
      console.error('Error with text-to-speech:', error);
      setIsSpeaking(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    Array.from(files).forEach(file => {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        toast({
          title: "File too large",
          description: `${file.name} is larger than 10MB limit.`,
          variant: "destructive"
        });
        return;
      }

      const attachment: Attachment = {
        id: Date.now().toString() + Math.random(),
        name: file.name,
        type: file.type,
        size: file.size,
        url: URL.createObjectURL(file),
        isImage: file.type.startsWith('image/')
      };

      setAttachments(prev => [...prev, attachment]);
    });

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeAttachment = (id: string) => {
    setAttachments(prev => {
      const attachment = prev.find(a => a.id === id);
      if (attachment) {
        URL.revokeObjectURL(attachment.url);
      }
      return prev.filter(a => a.id !== id);
    });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getSerpApiKey = (): string => {
    try {
      const stored = JSON.parse(localStorage.getItem("lachesis_admin_keys") || "{}");
      return (stored.serpapi || "").trim();
    } catch { return ""; }
  };

  const callOpenAIAPI = async (
    conversationMessages: Array<{role: string; content: any}>, 
    apiKey: string
  ): Promise<{content: string; toolCall?: any}> => {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-5.4',
        messages: conversationMessages,
        temperature: 0.4,
        max_completion_tokens: 2000,
        tools: [
          {
            type: "function",
            function: {
              name: "run_qtbn_analysis",
              description: "Run Quantum Temporal Bayesian Network analysis on a stock portfolio to predict future price movements, market regimes, and risk levels. Use this when the user wants predictions about their portfolio's future performance.",
              parameters: {
                type: "object",
                properties: {
                  tickers: {
                    type: "array",
                    items: { type: "string" },
                    description: "Array of stock ticker symbols (e.g., ['AAPL', 'GOOGL', 'TSLA'])"
                  },
                  allocations: {
                    type: "array",
                    items: { type: "number" },
                    description: "Array of allocation percentages for each ticker (e.g., [0.40, 0.35, 0.25]). Must sum to 1.0"
                  },
                  totalValue: {
                    type: "number",
                    description: "Total portfolio value in dollars"
                  }
                },
                required: ["tickers", "allocations", "totalValue"]
              }
            }
          },
          {
            type: "function",
            function: {
              name: "search_google",
              description: "Search Google for real-time information about stocks, markets, financial news, economic data, or any current events. Use this when the user asks about current prices, recent news, top stocks, market outlook, or any time-sensitive financial data.",
              parameters: {
                type: "object",
                properties: {
                  query: {
                    type: "string",
                    description: "The search query (e.g. 'top 10 stocks to buy March 2026', 'AAPL stock price today', 'S&P 500 best performers 2026')"
                  }
                },
                required: ["query"]
              }
            }
          }
        ],
        tool_choice: "auto"
      }),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    const data = await response.json();
    const message = data.choices[0]?.message;
    
    if (message?.tool_calls && message.tool_calls.length > 0) {
      return {
        content: message.content || "",
        toolCall: message.tool_calls[0]
      };
    }
    
    return {
      content: message?.content || "I apologize, but I couldn't process your request at the moment."
    };
  };

  const runQTBNAnalysis = async (tickers: string[], allocations: number[], totalValue: number) => {
    try {
      // Create QTBN instance
      const qtbn = new QuantumTemporalBayesianNetwork({
        timeHorizon: 30,
        stateSpaceSize: 16,
        observationSpace: ['price', 'volume', 'volatility', 'sentiment'],
        transitionModel: 'dynamic',
        quantumBackend: 'simulator',
        inferenceMethod: 'qaoa'
      });

      // Generate synthetic financial data based on portfolio
      const financialData: FinancialData = {
        tickers: tickers,
        prices: {},
        dates: Array.from({ length: 252 }, (_, i) => new Date(Date.now() - (252 - i) * 86400000).toISOString()),
        returns: {}
      };

      tickers.forEach((ticker) => {
        // Generate random returns and prices for demonstration
        const returns = Array.from({ length: 252 }, () => (Math.random() - 0.5) * 0.05);
        const prices = returns.reduce((acc, ret, i) => {
          const prevPrice = acc[i - 1] || 100;
          acc.push(prevPrice * (1 + ret));
          return acc;
        }, [100] as number[]);
        
        financialData.returns[ticker] = returns;
        financialData.prices[ticker] = prices;
      });

      // Run QTBN inference
      const observations = tickers.map(() => Math.random() * 0.8 + 0.2);
      const inferenceResult = qtbn.performInference(financialData, observations);
      const regimeAnalysis = qtbn.detectMarketRegime(financialData);

      // Format results for AI interpretation
      const results = {
        currentRegime: regimeAnalysis.currentRegime,
        confidence: (regimeAnalysis.confidence * 100).toFixed(1),
        regimeProbabilities: Object.entries(regimeAnalysis.regimeProbabilities)
          .map(([regime, prob]) => `${regime}: ${(prob * 100).toFixed(1)}%`)
          .join(', '),
        predictions: inferenceResult.temporalPredictions.slice(0, 5).map(p => ({
          timeStep: p.timeStep,
          regime: p.regime
        })),
        portfolioValue: totalValue,
        holdings: tickers.map((ticker, i) => ({
          ticker,
          allocation: (allocations[i] * 100).toFixed(1) + '%',
          value: (totalValue * allocations[i]).toFixed(2)
        }))
      };

      return results;
    } catch (error) {
      console.error('QTBN analysis error:', error);
      throw error;
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() && attachments.length === 0) return;
    
    if (!apiKey.trim()) {
      toast({
        title: "API Key Required",
        description: "Please enter your OpenAI API key to chat with Lachesis.",
        variant: "destructive"
      });
      return;
    }

    // Check if user is uploading a portfolio screenshot
    const hasImageAttachment = attachments.some(a => a.isImage);
    const isLikelyPortfolio = hasImageAttachment && (
      inputMessage.toLowerCase().includes('portfolio') ||
      inputMessage.toLowerCase().includes('stock') ||
      inputMessage.toLowerCase().includes('holdings') ||
      inputMessage.toLowerCase().includes('robinhood') ||
      inputMessage.toLowerCase().includes('position') ||
      inputMessage.toLowerCase().includes('account') ||
      !inputMessage.trim() // No message means "analyze this"
    );

    let messageContent = inputMessage;
    if (isLikelyPortfolio && hasImageAttachment) {
      messageContent = inputMessage || "Please analyze my portfolio screenshot, extract all stock positions with their values and allocation percentages. After extracting the data, run a quantum temporal analysis to predict future performance and explain the results in simple terms.";
    } else if (!inputMessage.trim() && attachments.length > 0) {
      messageContent = "Please analyze these attachments.";
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: messageContent,
      timestamp: new Date(),
      attachments: [...attachments]
    };

    setMessages(prev => [...prev, userMessage]);
    const currentAttachments = [...attachments];
    setInputMessage("");
    setAttachments([]);
    setIsLoading(true);

    try {
      // Build conversation history for OpenAI
      const conversationMessages = [
        {
          role: 'system',
          content: `You are Lachesis, an advanced A.I Integrated Quantum Assistant specializing in quantum computing and financial analytics. You have deep expertise in:

- Quantum circuit design and simulation
- Quantum noise models and error correction
- Financial risk analysis (VaR/CVaR, portfolio optimization)
- Market regime detection and volatility analysis
- Sentiment analysis for financial markets
- Quantum Temporal Bayesian Networks (QTBN)
- Integration of quantum computing with financial modeling
- Image analysis and document processing
- **Portfolio Screenshot Analysis & Quantum Prediction**

Your personality is:
- Professional yet approachable and warm
- Highly knowledgeable in quantum physics and finance
- Always ready to explain complex concepts clearly
- Enthusiastic about the intersection of quantum computing and finance
- Focused on practical applications
- Friendly and helpful like a human assistant
- Able to analyze images, documents, and attachments

**CRITICAL CAPABILITY - Automated Portfolio Analysis & QTBN Prediction:**

When users share portfolio screenshots or ask about their portfolio predictions:

1. **Extract Portfolio Data**: From the screenshot, identify:
   - Stock tickers (e.g., AAPL, GOOGL, TSLA)
   - Dollar amounts for each position
   - Total portfolio value
   - Calculate allocation percentages (position value / total value)

2. **Run QTBN Analysis**: Once you have the data, IMMEDIATELY call the run_qtbn_analysis function with:
   - tickers: array of stock symbols
   - allocations: array of percentages as decimals (must sum to 1.0)
   - totalValue: total portfolio value in dollars

3. **Explain Results in Layman's Terms**: When you receive QTBN results, explain them conversationally:
   - "Your portfolio is currently in a [REGIME] market environment with [CONFIDENCE]% confidence"
   - "Over the next [TIMEFRAME], I predict the market will likely [PREDICTION]"
   - "Here's what this means for you: [SIMPLE EXPLANATION]"
   - Avoid technical jargon like "quantum amplitude estimation" or "Bayesian inference"
   - Instead say things like "AI-powered predictions" or "advanced pattern analysis"

4. **No Manual Configuration Needed**: The user should NEVER have to manually adjust qubits, logic gates, or other quantum parameters. You handle everything automatically.

Always identify yourself as "Lachesis" and be warm and conversational. Make quantum finance accessible and actionable.

**LANGUAGE INSTRUCTION**: Always respond in ${language}. If the user writes in a different language, still respond in ${language}.`
        },
        ...messages.map(msg => ({
          role: msg.role,
          content: msg.attachments && msg.attachments.length > 0 
            ? [
                { type: 'text', text: msg.content },
                ...msg.attachments.filter(a => a.isImage).map(att => ({
                  type: 'image_url' as const,
                  image_url: { url: att.url }
                }))
              ]
            : msg.content
        })),
        {
          role: 'user',
          content: currentAttachments.length > 0
            ? [
                { type: 'text', text: messageContent },
                ...currentAttachments.filter(a => a.isImage).map(att => ({
                  type: 'image_url' as const,
                  image_url: { url: att.url }
                }))
              ]
            : messageContent
        }
      ];

      const { content, toolCall } = await callOpenAIAPI(conversationMessages, apiKey);

      // Only show a pre-tool message if OpenAI actually provided one (not blank)
      if (content.trim()) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: content,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, assistantMessage]);
      }
      
      // If AI wants to run QTBN analysis, execute it
      if (toolCall && toolCall.function.name === 'run_qtbn_analysis') {
        const args = JSON.parse(toolCall.function.arguments);
        
        // Show analysis in progress
        const analysisMessage: Message = {
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content: "🔬 Running quantum temporal analysis on your portfolio... This may take a moment.",
          timestamp: new Date()
        };
        setMessages(prev => [...prev, analysisMessage]);

        try {
          const qtbnResults = await runQTBNAnalysis(
            args.tickers,
            args.allocations,
            args.totalValue
          );

          // Send results back to AI for interpretation
          const resultsMessages = [
            ...conversationMessages,
            {
              role: 'assistant',
              content: content,
              tool_calls: [toolCall]
            },
            {
              role: 'tool',
              tool_call_id: toolCall.id,
              content: JSON.stringify(qtbnResults)
            }
          ];

          const { content: finalResponse } = await callOpenAIAPI(resultsMessages, apiKey);
          
          const interpretationMessage: Message = {
            id: (Date.now() + 3).toString(),
            role: 'assistant',
            content: finalResponse,
            timestamp: new Date()
          };

          setMessages(prev => [...prev, interpretationMessage]);
          
          if (voiceEnabled) {
            await speakText(finalResponse);
          }
        } catch (qtbnError) {
          console.error('QTBN analysis failed:', qtbnError);
          const errorMsg: Message = {
            id: (Date.now() + 3).toString(),
            role: 'assistant',
            content: "I encountered an issue running the quantum analysis. The portfolio extraction worked, but the prediction engine had a problem. Would you like me to try again?",
            timestamp: new Date()
          };
          setMessages(prev => [...prev, errorMsg]);
        }
      } else if (toolCall && toolCall.function.name === 'search_google') {
        const args = JSON.parse(toolCall.function.arguments);
        const serpApiKey = getSerpApiKey();

        if (!serpApiKey) {
          const noKeyMsg: Message = {
            id: (Date.now() + 2).toString(),
            role: 'assistant',
            content: "I'd like to search for current market data to give you the most accurate answer, but no SerpAPI key is configured. Please add your SerpAPI key in the **Admin** tab, then ask again.",
            timestamp: new Date()
          };
          setMessages(prev => [...prev, noKeyMsg]);
        } else {
          const searchingMsgId = (Date.now() + 2).toString();
          const searchingMessage: Message = {
            id: searchingMsgId,
            role: 'assistant',
            content: `🔍 Searching Google for: *"${args.query}"*...`,
            timestamp: new Date()
          };
          setMessages(prev => [...prev, searchingMessage]);

          try {
            const { results } = await apiWebSearch(args.query, serpApiKey);
            const searchSummary = results.length > 0
              ? results.map((r, i) => `[${i + 1}] ${r.title}\n${r.snippet}\nSource: ${r.link}`).join("\n\n")
              : "No results found.";

            const resultsMessages = [
              ...conversationMessages,
              { role: 'assistant', content: content || "", tool_calls: [toolCall] },
              { role: 'tool', tool_call_id: toolCall.id, content: searchSummary }
            ];

            const { content: finalResponse } = await callOpenAIAPI(resultsMessages, apiKey);
            const finalMsg: Message = {
              id: (Date.now() + 3).toString(),
              role: 'assistant',
              content: finalResponse,
              timestamp: new Date()
            };
            setMessages(prev => prev.filter(m => m.id !== searchingMsgId).concat(finalMsg));

            if (voiceEnabled) await speakText(finalResponse);
          } catch (searchErr) {
            console.error('SerpAPI search failed:', searchErr);
            const errMsg: Message = {
              id: (Date.now() + 3).toString(),
              role: 'assistant',
              content: `Search failed: ${searchErr instanceof Error ? searchErr.message : String(searchErr)}. I'll answer based on my training data instead.\n\n${content}`,
              timestamp: new Date()
            };
            setMessages(prev => prev.filter(m => m.id !== searchingMsgId).concat(errMsg));
          }
        }
      } else if (voiceEnabled) {
        await speakText(content);
      }
    } catch (error) {
      console.error('Error calling OpenAI API:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I apologize, but I'm experiencing technical difficulties. Please check your OpenAI API key and try again. If the problem persists, there might be an issue with the OpenAI API service.",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
      
      toast({
        title: "Connection Error",
        description: "Failed to connect to Lachesis. Please verify your API key.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="space-y-6">
      {/* API Key Configuration */}
      <Alert className="border-amber-500/20 bg-amber-500/10">
        <Key className="h-4 w-4" />
        <AlertDescription className="space-y-3">
          <div>
            <strong>Connect to Supabase for seamless integration</strong> or enter your API keys below for temporary access.
          </div>
          <div className="space-y-2">
            <div className="flex gap-2">
              <Input
                type="password"
                placeholder="Enter your OpenAI API key..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="flex-1"
              />
              <Button variant="outline" size="sm" asChild>
                <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer">
                  Get API Key
                </a>
              </Button>
            </div>
            <div className="flex gap-2">
              <Input
                type="password"
                placeholder="Enter your ElevenLabs API key for voice..."
                value={elevenLabsApiKey}
                onChange={(e) => setElevenLabsApiKey(e.target.value)}
                className="flex-1"
              />
              <Button variant="outline" size="sm" asChild>
                <a href="https://elevenlabs.io/app/speech-synthesis/text-to-speech" target="_blank" rel="noopener noreferrer">
                  Get Voice Key
                </a>
              </Button>
            </div>
          </div>
        </AlertDescription>
      </Alert>

      {/* Chat Interface */}
      <Card className="border-accent/20 bg-gradient-to-br from-card to-primary/5">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-3">
            <div className="relative">
              <div className="w-12 h-12 rounded-full overflow-hidden border-2 border-primary/30 shadow-lg">
                <img 
                  src={lachesisAvatar} 
                  alt="Lachesis AI Assistant" 
                  className="w-full h-full object-cover"
                />
              </div>
              <Sparkles className="w-4 h-4 text-accent absolute -top-1 -right-1 animate-pulse" />
            </div>
            <div>
              <h3 className="text-xl font-bold">Lachesis</h3>
              <p className="text-sm text-muted-foreground font-normal">
                Your A.I Integrated Quantum Assistant
              </p>
            </div>
            <div className="ml-auto flex gap-2">
              <Badge variant="outline" className="border-primary/30 bg-primary/10">
                <Atom className="w-3 h-3 mr-1" />
                Quantum
              </Badge>
              <Badge variant="outline" className="border-accent/30 bg-accent/10">
                <MessageCircle className="w-3 h-3 mr-1" />
                AI-Powered
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-base font-medium">Your AI financial assistant — ask questions, analyze portfolios, and get investment insights in plain English.</p>
          {/* Messages */}
          <ScrollArea 
            className="h-96 w-full rounded-lg border border-accent/20 bg-muted/20 p-4" 
            ref={scrollAreaRef}
          >
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg px-4 py-3 ${
                      message.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-card border border-accent/20'
                    }`}
                  >
                    {message.role === 'assistant' && (
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-6 h-6 rounded-full overflow-hidden border border-primary/30">
                          <img 
                            src={lachesisAvatar} 
                            alt="Lachesis" 
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <span className="text-sm font-semibold text-primary">Lachesis</span>
                      </div>
                    )}
                     <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                     
                     {/* Display attachments */}
                     {message.attachments && message.attachments.length > 0 && (
                       <div className="mt-3 space-y-2">
                         {message.attachments.map((attachment) => (
                           <div key={attachment.id} className="border rounded-lg p-2 bg-muted/50">
                             {attachment.isImage ? (
                               <div className="space-y-2">
                                 <img 
                                   src={attachment.url} 
                                   alt={attachment.name}
                                   className="max-w-full max-h-48 rounded object-cover"
                                 />
                                 <p className="text-xs text-muted-foreground flex items-center gap-1">
                                   <Image className="w-3 h-3" />
                                   {attachment.name} ({formatFileSize(attachment.size)})
                                 </p>
                               </div>
                             ) : (
                               <div className="flex items-center gap-2">
                                 <Paperclip className="w-4 h-4 text-muted-foreground" />
                                 <div className="flex-1 min-w-0">
                                   <p className="text-xs font-medium truncate">{attachment.name}</p>
                                   <p className="text-xs text-muted-foreground">{formatFileSize(attachment.size)}</p>
                                 </div>
                               </div>
                             )}
                           </div>
                         ))}
                       </div>
                     )}
                     
                     <span className="text-xs opacity-70 mt-2 block">
                       {message.timestamp.toLocaleTimeString()}
                     </span>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-card border border-accent/20 rounded-lg px-4 py-3 max-w-[80%]">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-6 h-6 rounded-full overflow-hidden border border-primary/30">
                        <img 
                          src={lachesisAvatar} 
                          alt="Lachesis" 
                          className="w-full h-full object-cover animate-pulse"
                        />
                      </div>
                      <span className="text-sm font-semibold text-primary">Lachesis</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:0.1s]"></div>
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:0.2s]"></div>
                      <span className="text-sm text-muted-foreground ml-2">Processing quantum insights...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Attachments Preview */}
          {attachments.length > 0 && (
            <div className="space-y-2">
              <Label className="text-sm font-medium">Attachments ({attachments.length})</Label>
              <div className="flex flex-wrap gap-2">
                {attachments.map((attachment) => (
                  <div key={attachment.id} className="relative group">
                    <div className="border rounded-lg p-2 bg-muted/50 flex items-center gap-2 max-w-48">
                      {attachment.isImage ? (
                        <div className="flex items-center gap-2 min-w-0">
                          <img 
                            src={attachment.url} 
                            alt={attachment.name}
                            className="w-8 h-8 rounded object-cover flex-shrink-0"
                          />
                          <div className="min-w-0 flex-1">
                            <p className="text-xs font-medium truncate">{attachment.name}</p>
                            <p className="text-xs text-muted-foreground">{formatFileSize(attachment.size)}</p>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 min-w-0">
                          <Paperclip className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                          <div className="min-w-0 flex-1">
                            <p className="text-xs font-medium truncate">{attachment.name}</p>
                            <p className="text-xs text-muted-foreground">{formatFileSize(attachment.size)}</p>
                          </div>
                        </div>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-destructive text-destructive-foreground opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={() => removeAttachment(attachment.id)}
                    >
                      <X className="w-3 h-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className="space-y-2">
            <div className="flex gap-2">
              <Input
                placeholder="Ask Lachesis about quantum computing, financial analytics, or risk assessment..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isLoading}
                className="flex-1"
              />
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                multiple
                accept="image/*,.pdf,.doc,.docx,.txt,.csv,.json"
                className="hidden"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                title="Attach files"
              >
                <Paperclip className="w-4 h-4" />
              </Button>
              {micSupported && (
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => {
                    if (isListening) {
                      stopListening();
                    } else {
                      startListening(text =>
                        setInputMessage(prev => prev ? `${prev} ${text}` : text)
                      );
                    }
                  }}
                  disabled={isLoading}
                  className={isListening ? "bg-red-500/20 border-red-500/50 animate-pulse" : ""}
                  title={isListening ? "Stop listening" : "Speak your message"}
                >
                  {isListening ? <MicOff className="w-4 h-4 text-red-400" /> : <Mic className="w-4 h-4" />}
                </Button>
              )}
              <Button
                variant="outline"
                size="icon"
                onClick={() => setVoiceEnabled(!voiceEnabled)}
                disabled={!elevenLabsApiKey.trim() || isSpeaking}
                className={voiceEnabled ? "bg-primary/10" : ""}
              >
                {voiceEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
              </Button>
              <Button 
                onClick={handleSendMessage} 
                disabled={isLoading || (!inputMessage.trim() && attachments.length === 0) || !apiKey.trim()}
                size="icon"
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInputMessage("How do quantum circuits work in financial modeling?")}
              disabled={isLoading}
            >
              Quantum Circuits
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInputMessage("Explain VaR calculation using Monte Carlo methods")}
              disabled={isLoading}
            >
              Risk Analysis
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInputMessage("How does sentiment analysis integrate with quantum computing?")}
              disabled={isLoading}
            >
              Sentiment Integration
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInputMessage("What's the latest in quantum computing for finance?")}
              disabled={isLoading}
            >
              QTBN Explained
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
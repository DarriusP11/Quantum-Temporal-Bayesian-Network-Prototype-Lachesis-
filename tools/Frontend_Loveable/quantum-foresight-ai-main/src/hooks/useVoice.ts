/**
 * useVoice.ts — Browser Web Speech API hook.
 * Provides microphone input (SpeechRecognition) and TTS output (SpeechSynthesis).
 * No API key required — uses native browser capabilities.
 *
 * SpeechRecognition: Chrome, Edge, Safari 15+  (requires HTTPS or localhost)
 * SpeechSynthesis:   all modern browsers
 */
import { useState, useRef, useCallback } from "react";

// Maps app language names (from AppContext) → BCP 47 language tags
const LANG_CODES: Record<string, string> = {
  "English":                "en-US",
  "Spanish":                "es-ES",
  "French":                 "fr-FR",
  "German":                 "de-DE",
  "Italian":                "it-IT",
  "Portuguese":             "pt-BR",
  "Dutch":                  "nl-NL",
  "Russian":                "ru-RU",
  "Chinese (Simplified)":   "zh-CN",
  "Chinese (Traditional)":  "zh-TW",
  "Japanese":               "ja-JP",
  "Korean":                 "ko-KR",
  "Arabic":                 "ar-SA",
  "Hindi":                  "hi-IN",
  "Bengali":                "bn-IN",
  "Turkish":                "tr-TR",
  "Vietnamese":             "vi-VN",
  "Thai":                   "th-TH",
  "Indonesian":             "id-ID",
  "Malay":                  "ms-MY",
  "Polish":                 "pl-PL",
  "Swedish":                "sv-SE",
  "Norwegian":              "nb-NO",
  "Danish":                 "da-DK",
  "Finnish":                "fi-FI",
  "Romanian":               "ro-RO",
  "Greek":                  "el-GR",
  "Hungarian":              "hu-HU",
  "Czech":                  "cs-CZ",
  "Slovak":                 "sk-SK",
  "Ukrainian":              "uk-UA",
  "Bulgarian":              "bg-BG",
  "Croatian":               "hr-HR",
  "Serbian":                "sr-RS",
  "Lithuanian":             "lt-LT",
  "Latvian":                "lv-LV",
  "Estonian":               "et-EE",
  "Hebrew":                 "he-IL",
  "Persian":                "fa-IR",
  "Urdu":                   "ur-PK",
  "Swahili":                "sw-KE",
  "Tamil":                  "ta-IN",
  "Telugu":                 "te-IN",
  "Kannada":                "kn-IN",
  "Marathi":                "mr-IN",
  "Gujarati":               "gu-IN",
  "Punjabi":                "pa-IN",
  "Afrikaans":              "af-ZA",
  "Catalan":                "ca-ES",
  "Welsh":                  "cy-GB",
};

export function useVoice(appLanguage = "English") {
  const langCode = LANG_CODES[appLanguage] ?? "en-US";

  const [isListening, setIsListening]  = useState(false);
  const [isSpeaking,  setIsSpeaking]   = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);

  const micSupported =
    typeof window !== "undefined" &&
    ("SpeechRecognition" in window || "webkitSpeechRecognition" in window);

  const ttsSupported =
    typeof window !== "undefined" && "speechSynthesis" in window;

  /** Start mic listening. Calls onResult(transcript) when the user stops speaking. */
  const startListening = useCallback(
    (onResult: (text: string) => void) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const SR = (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition;
      if (!SR) return;

      const rec = new SR();
      rec.lang = langCode;
      rec.continuous = false;
      rec.interimResults = false;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      rec.onresult = (e: any) => {
        const text: string = e.results[0][0].transcript;
        onResult(text);
      };
      rec.onend  = () => setIsListening(false);
      rec.onerror = () => setIsListening(false);

      recognitionRef.current = rec;
      rec.start();
      setIsListening(true);
    },
    [langCode],
  );

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    setIsListening(false);
  }, []);

  /** Speak text aloud using browser TTS (SpeechSynthesis). */
  const speak = useCallback(
    (text: string) => {
      if (!window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      const utt = new SpeechSynthesisUtterance(text);
      utt.lang = langCode;
      utt.onstart = () => setIsSpeaking(true);
      utt.onend   = () => setIsSpeaking(false);
      utt.onerror = () => setIsSpeaking(false);
      window.speechSynthesis.speak(utt);
    },
    [langCode],
  );

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis?.cancel();
    setIsSpeaking(false);
  }, []);

  return {
    isListening, isSpeaking,
    micSupported, ttsSupported,
    startListening, stopListening,
    speak, stopSpeaking,
  };
}

import { useCallback, useEffect, useRef, useState } from "react";

interface Options {
  onInterim: (text: string) => void;   // live partial transcript
  onFinal:   (text: string) => void;   // confirmed word(s) to append
  onError:   (msg: string)  => void;
}

export function useSpeechRecognition({ onInterim, onFinal, onError }: Options) {
  const [listening, setListening] = useState(false);
  const recogRef   = useRef<SpeechRecognition | null>(null);
  const activeRef  = useRef(false);               // survive closure captures

  const supported =
    typeof window !== "undefined" &&
    ("SpeechRecognition" in window || "webkitSpeechRecognition" in window);

  const start = useCallback(() => {
    if (!supported) { onError("Speech recognition is not supported in this browser. Use Chrome or Edge."); return; }
    if (activeRef.current) return;

    const SR = (window.SpeechRecognition ?? (window as any).webkitSpeechRecognition) as typeof SpeechRecognition;
    const rec = new SR();
    rec.continuous      = true;
    rec.interimResults  = true;
    rec.lang            = "en-US";

    rec.onstart = () => { activeRef.current = true; setListening(true); };
    rec.onend   = () => { activeRef.current = false; setListening(false); onInterim(""); };

    rec.onresult = (e: SpeechRecognitionEvent) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) {
          onFinal(t);
        } else {
          interim += t;
        }
      }
      onInterim(interim);
    };

    rec.onerror = (e: SpeechRecognitionErrorEvent) => {
      onError(`Microphone error: ${e.error}`);
      activeRef.current = false;
      setListening(false);
    };

    recogRef.current = rec;
    rec.start();
  }, [supported, onInterim, onFinal, onError]);

  const stop = useCallback(() => {
    recogRef.current?.stop();
    recogRef.current = null;
  }, []);

  // clean up on unmount
  useEffect(() => () => { recogRef.current?.abort(); }, []);

  return { listening, supported, start, stop };
}

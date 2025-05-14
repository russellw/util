// app/page.tsx
import TemperatureConverter from '@/components/TemperatureConverter'; // Adjust path if needed

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-12 bg-gray-100">
      <TemperatureConverter />
    </main>
  );
}
import { Component, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  template: `
    <div style="text-align:center; margin-top:100px; font-family:Arial;">
      <h1>🎙️ Meeting-to-Action AI</h1>
      <p style="font-size:24px; color:#2563eb;">
        {{ message() }}
      </p>
    </div>
  `
})
export class App {
  private http = inject(HttpClient);

  message = signal('Connecting to backend...');

  constructor() {
    this.http.get<any>('http://localhost:8000/')
      .subscribe({
        next: (res) => this.message.set(res.message),
        error: () => this.message.set('Backend connection failed ❌')
      });
  }
}
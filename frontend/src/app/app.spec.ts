import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { App } from './app';

describe('App', () => {
  let httpTestingController: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    }).compileComponents();

    httpTestingController = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpTestingController.verify();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should render the Meeting-to-Action AI title', async () => {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('Meeting-to-Action AI');
  });

  it('should initialize with idle state and empty results', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app.processingStage()).toBe('idle');
    expect(app.uploadProgress()).toBe(0);
    expect(app.transcript()).toBe('');
    expect(app.summary()).toBe('');
    expect(app.keyPoints().length).toBe(0);
    expect(app.decisions().length).toBe(0);
    expect(app.actionItems().length).toBe(0);
  });

  it('should toggle action item completed state', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    app.actionItems.set([
      { task: 'Test action', assignee: 'Team', deadline: 'Friday', priority: 'High', completed: false }
    ]);
    expect(app.actionItems()[0].completed).toBe(false);

    app.toggleActionItem(0);
    expect(app.actionItems()[0].completed).toBe(true);

    app.toggleActionItem(0);
    expect(app.actionItems()[0].completed).toBe(false);
  });

  it('should filter transcript segments by search query', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    app.segments.set([
      { timestamp: '00:01', start: 1, end: 5, text: 'Discussing the backend API' },
      { timestamp: '00:15', start: 15, end: 20, text: 'Designing the Angular UI' }
    ]);

    expect(app.filteredSegments().length).toBe(2);

    app.searchQuery.set('API');
    expect(app.filteredSegments().length).toBe(1);
    expect(app.filteredSegments()[0].text).toContain('backend API');

    app.searchQuery.set('NonExistent');
    expect(app.filteredSegments().length).toBe(0);
  });

  it('should format file sizes properly', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app.formatFileSize(0)).toBe('0 B');
    expect(app.formatFileSize(1024)).toBe('1 KB');
    expect(app.formatFileSize(1024 * 1024 * 5)).toBe('5 MB');
  });
});
